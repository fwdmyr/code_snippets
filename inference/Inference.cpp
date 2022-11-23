//
// Created by felix on 20.07.22.
//

#include "inference/Inference.h"

namespace fgo::inference {


    InferenceEngine::InferenceEngine(InferenceConfig config) : config_(std::move(config)), rosMutex_(), numRegisteredSatellitesLastRun_(0) {
        systemClockPtr_ = std::make_shared<rclcpp::Clock>();
        logger_ = std::make_unique<rclcpp::Logger>(rclcpp::get_logger(config_.name));
        registeredSatelliteIds_.reserve(fgo::constants::maxSatellites);
        initTypeNamesMap();
        initTopicNames();
    }

    void InferenceEngine::run() {
        /*
         * Inference should be performed for each factor registered with the inference engine
         * If we want to use Gaussian noise model, we do not register the factor with the inference engine
         * Residuals class holds a live GMM object that gets fit to the residual data.
         * After convergence or max iterations, cached GMM object gets updated with live GMM.
         * The InferenceEngine class only provides (thread-safe) access to the cached GMM. At the cost of a possibly
         * slightly outdated noise model, the FGO loop never has to wait for the inference loop
         * Thread Logic:
         * For each registered satellite
         *    For each registered factor of satellite
         *        // Buffers store fixed amount of residuals. If satellite went out of sight and reappeared recently,
         *        // some residuals in buffer are too old to inference current noise distribution
         *        If sufficient amount of recent residuals available (N samples in the interval [t_now - inference_lag, t_now])
         *            // Avoid starving of last registered factors
         *            If enough time has passed since last inference
         *                Run inference for the factor
         *                Update the cached GMM
         */

        // Give the FGO some time to setup since we need to aquire some residual data for inference anyway
        if (registeredSatelliteIds_.empty())
            std::this_thread::sleep_for(std::chrono::nanoseconds(config_.waitTimeBetweenRuns.nanoseconds()));
        // Loop over all satellites that were encountered so far
        for (const auto &satelliteId: registeredSatelliteIds_) {
            const auto satellite = satellitesArray_[satelliteId];
            if (!satellite)
                continue;
            // Loop over all factor types associated with satellite that were encountered so far
            for (const auto &factorType: satellite->getRegisteredFactorTypes()) {
                const auto [ec, elapsedTime] = satellite->runInference(factorType);
                const auto topicName = topicNamesPtr_->at(factorType);
                if (ec == ERRORCODE::SUCCESS) {
                    RCLCPP_INFO_STREAM(*logger_,
                                       "[Satellite " << satelliteId << " | Factor " << topicName
                                                     << "]: Inference finished in " << elapsedTime << " ms");

                    if (gmmPublisher_) {
                        auto mixtureMsg = satellite->buildMixtureMessage(factorType, elapsedTime);
                        mixtureMsg.header.frame_id = config_.name;
                        gmmPublisher_->publish(mixtureMsg);
                    }
                }
            }
        }
    }

    void InferenceEngine::threadedRun() {

        /*
         * Inference should be performed for each factor registered with the inference engine
         * If we want to use Gaussian noise model, we do not register the factor with the inference engine
         * Residuals class holds a live GMM object that gets fit to the residual data.
         * After convergence or max iterations, cached GMM object gets updated with live GMM.
         * The InferenceEngine class only provides (thread-safe) access to the cached GMM. At the cost of a possibly
         * slightly outdated noise model, the FGO loop never has to wait for the inference loop
         * Thread Logic:
         * For each registered satellite
         *    For each registered factor of satellite
         *        // Buffers store fixed amount of residuals. If satellite went out of sight and reappeared recently,
         *        // some residuals in buffer are too old to inference current noise distribution
         *        If sufficient amount of recent residuals available (N samples in the interval [t_now - inference_lag, t_now])
         *            // Avoid starving of last registered factors
         *            If enough time has passed since last inference
         *                Run inference for the factor
         *                Update the cached GMM
         */

        // Give the FGO some time to setup since we need to aquire some residual data for inference anyway
        if (registeredSatelliteIds_.empty())
            std::this_thread::sleep_for(std::chrono::nanoseconds(config_.waitTimeBetweenRuns.nanoseconds()));
        if (registeredSatelliteIds_.size() == numRegisteredSatellitesLastRun_)
            return;
        numRegisteredSatellitesLastRun_ = registeredSatelliteIds_.size();
        // Loop over all satellites that were encountered so far
        for (const auto &satelliteId: registeredSatelliteIds_) {
            const auto satellite = satellitesArray_[satelliteId];
            // If satellite was registered but no satellite object constructed yet
            // This scenario would be caused if data race occurs
            if (!satellite)
                continue;
            // If satellite appears for the first time, array entry is standard-constructed nullptr which we can check against
            if (!inferenceThreads_[satelliteId]) {
                RCLCPP_INFO_STREAM(*logger_, "Starting inference thread for satellite " << satelliteId);
                auto inferenceThread = std::make_unique<std::thread>([this, satelliteId, satellite]() -> void {
                    while (rclcpp::ok()) {

                        std::unique_lock<std::mutex> tlg(inferenceMutexes_[satelliteId]);
                        inferenceCVs_[satelliteId].wait(tlg, [&] { return inferenceTriggers_[satelliteId]; });

                        // Loop over all factor types associated with satellite that were encountered so far
                        for (const auto &factorType: satellite->getRegisteredFactorTypes()) {
                            // Call thread-safe runInference since stallite instantiates a self-contained class
                            const auto [ec, elapsedTime] = satellite->runInference(factorType);
                            // Simultaneous reads are thread-safe
                            const auto topicName = topicNamesPtr_->at(factorType);
                            if (ec == ERRORCODE::SUCCESS) {
                                // Lock the access to the ROS node and publisher
                                std::lock_guard<std::mutex> lg(rosMutex_);
                                RCLCPP_INFO_STREAM(*logger_,
                                                   "[Satellite " << satelliteId << " | Factor " << topicName
                                                                 << "]: Inference finished in " << elapsedTime
                                                                 << " ms");

                                if (gmmPublisher_) {
                                    auto mixtureMsg = satellite->buildMixtureMessage(factorType, elapsedTime);
                                    mixtureMsg.header.frame_id = config_.name;
                                    gmmPublisher_->publish(mixtureMsg);
                                }
                                // Lock-guard goes out of scope here
                            }
                        }
                        inferenceTriggers_[satelliteId] = false;
                    }
                });
                // Move the ptr to the thread in the container after thread was properly constructed
                inferenceThreads_[satelliteId] = std::move(inferenceThread);
            }
        }
        RCLCPP_INFO_STREAM(*logger_,
                           "Now running inference in " << numRegisteredSatellitesLastRun_ << " threads");
    }

    void InferenceEngine::updateResidualData(const graphs::GraphBase &graph, const gtsam::Values &result,
                                             const rclcpp::Time &timestamp) {
        // Loop over all satellites that were encountered so far
        for (const auto &satelliteId: registeredSatelliteIds_) {
            const auto &satellite = satellitesArray_[satelliteId];
            if (!satellite)
                continue;
            // Loop over all factor types associated with satellite that were encountered so far
            for (const auto &factorType: satellite->getRegisteredFactorTypes()) {
                // For each GNSS factor in graph constructed for optimization that just completed:
                // - Retrieve the factor type enum value that we need for downcasting to specific derived factor
                // - Retrieve index where factor of factor type is stored in graph
                // - Get pointer to factor and its variable keys
                // - Build the values associated with keys from optimization result
                // - Compute the residual |h(x) - Z|
                const auto idxQueue = satellite->getIndexQueue(factorType);
                if (!idxQueue)
                    continue;
                // Get the residual data for each factor of factorType stored in graph at indexes held by idxQueue
                while (!idxQueue->empty()) {
                    gtsam::Vector residualData;
                    size_t idx = idxQueue->front();
                    idxQueue->pop();

                    // switch between templated functions to allow for downcasting to the correct derived factor
                    switch (factorType) {

                        case FACTOR::DDCP:
                            residualData = computeResidual<factors::DDCarrierPhaseFactor>(graph, result, idx);
                            break;
                        case FACTOR::DDPRDR:
                            residualData = computeResidual<factors::DDPRDRFactor>(graph, result, idx);
                            break;
                        case FACTOR::GPDDCP:
                            residualData = computeResidual<factors::GPInterpolatedDDCarPhaFactorO>(graph, result, idx);
                            break;
                        case FACTOR::GPDDPRDR:
                        case FACTOR::GPDDPR:
                            residualData = computeResidual<factors::GPInterpolatedDDPseuRaFactorO>(graph, result, idx);
                            break;
                        case FACTOR::GPPRDR:
                            residualData = computeResidual<factors::GPInterpolatedPRDRFactorO>(graph, result, idx);
                            break;
                        case FACTOR::GPTDCP:
                            residualData = computeResidual<factors::GPInterTDCPFactorO>(graph, result, idx);
                            break;
                        case FACTOR::GP3TDCP:
                            residualData = computeResidual<factors::GPInterpolated3TDCarPhaFactorO>(graph, result, idx);
                            break;
                        case FACTOR::PRDR:
                            residualData = computeResidual<factors::PRDRFactor>(graph, result, idx);
                            break;
                        case FACTOR::PR:
                            residualData = computeResidual<factors::PRFactor>(graph, result, idx);
                            break;
                        case FACTOR::PRGMM:
                            residualData = computeResidual<factors::PRFactorGMM>(graph, result, idx);
                            break;
                        case FACTOR::PRMEST:
                            residualData = computeResidual<factors::PRFactorMEst>(graph, result, idx);
                            break;
                        case FACTOR::DR:
                            residualData = computeResidual<factors::DRFactor>(graph, result, idx);
                            break;
                        case FACTOR::TDCP:
                            residualData = computeResidual<factors::TripleDiffCPFactor>(graph, result, idx);
                            break;
                        case FACTOR::TDCPLOCK:
                            residualData = computeResidual<factors::AmbiguityLockFactor>(graph, result, idx);
                            break;
                        case FACTOR::PR3:
                            residualData = computeResidual<factors::Pseudorange3>(graph, result, idx);
                            break;
                        case FACTOR::END:
                            throw std::runtime_error("updateResidualData: Invalid factor type");
                    }

                    if (!residualData.size())
                        continue;
                    // add residual data into buffers held by satellite
                    bool isOutlier = satellite->addResidual(factorType, timestamp, residualData);

                    // TODO: If inference thread is not running, trigger it here
                    inferenceTriggers_[satelliteId] = true;
                    inferenceCVs_[satelliteId].notify_one();

                    // if it is required, also publish the residual on its corresponding topic
                    if (resPublisher_) {
                        auto residualMsg = satellite->buildResidualMessage(factorType, timestamp, residualData, isOutlier);
                        residualMsg.header.frame_id = config_.name;
                        resPublisher_->publish(residualMsg);
                    }
                }
            }
        }
    }

    void InferenceEngine::setNode(const rclcpp::Node::SharedPtr &nodePtr, bool initPublishers) {
        nodePtr_ = nodePtr;
        if (initPublishers) {
            resPublisher_ = nodePtr_->create_publisher<irt_msgs::msg::Residual>("Residuals", 10);
            gmmPublisher_ = nodePtr_->create_publisher<irt_msgs::msg::GaussianMixture>("GaussianMixture", 10);
            satDataPublisher_ = nodePtr_->create_publisher<irt_msgs::msg::SatelliteData>("SatelliteData", 10);
            prFactorWeightsPublisher_ = nodePtr_->create_publisher<irt_msgs::msg::PRFactorWeights>("PRFactorWeights",
                                                                                                   10);
        }
        printConfig();
    }

    void InferenceEngine::publishPRFactorWeights(const std::map<uint32_t, gtsam::Vector1> &satIdWeightsMap) {
        irt_msgs::msg::PRFactorWeights msg;
        msg.header.stamp = nodePtr_->now();
        msg.header.frame_id = config_.name;
        std::for_each(satIdWeightsMap.cbegin(), satIdWeightsMap.cend(),
                      [&](const std::pair<const uint32_t, gtsam::Vector1> &entry) -> void {
                          msg.sat.push_back(entry.first);
                          msg.cardinality.push_back(1);
                          msg.weight.push_back(entry.second.value());
                      });
        msg.mean = fgo::utils::mean(msg.weight.begin(), msg.weight.end());
        msg.median = fgo::utils::median(msg.weight.begin(), msg.weight.end());
        msg.variance = fgo::utils::variance(msg.weight.begin(), msg.weight.end());
        prFactorWeightsPublisher_->publish(msg);
    }

    void InferenceEngine::publishPRFactorWeights(const std::map<uint32_t, fgo::data_types::DiscreteValue> &satIdWeightsMap) {
        irt_msgs::msg::PRFactorWeights msg;
        msg.header.stamp = nodePtr_->now();
        msg.header.frame_id = config_.name;
        std::for_each(satIdWeightsMap.cbegin(), satIdWeightsMap.cend(),
                      [&](const auto &entry) -> void {
                          msg.sat.push_back(entry.first);
                          msg.cardinality.push_back(entry.second.cardinality);
                          msg.weight.push_back(entry.second.value);
                      });
        msg.mean = fgo::utils::mean(msg.weight.begin(), msg.weight.end());
        msg.median = fgo::utils::median(msg.weight.begin(), msg.weight.end());
        msg.variance = fgo::utils::variance(msg.weight.begin(), msg.weight.end());
        prFactorWeightsPublisher_->publish(msg);
    }

    void InferenceEngine::printConfig() {
        RCLCPP_INFO_STREAM(*logger_, "Inference Engine initialized");

        RCLCPP_INFO_STREAM(*logger_, "[Strategy] Long Term: " << fgo::utils::bool2str(config_.longTerm));

        RCLCPP_INFO_STREAM(*logger_, "[Buffers] Buffer Size: " << config_.bufferSize);


        RCLCPP_INFO_STREAM(*logger_, "[Hyperparameters] Min Samples: " << config_.minSamples);
        RCLCPP_INFO_STREAM(*logger_, "[Hyperparameters] Window Size: "
                << fgo::constants::nanosec2sec * config_.windowSize.nanoseconds());
        RCLCPP_INFO_STREAM(*logger_, "[Hyperparameters] Wait Time: "
                << fgo::constants::nanosec2sec * config_.waitTimeBetweenRuns.nanoseconds());
        RCLCPP_INFO_STREAM(*logger_, "[Hyperparameters] Offset Time: "
                << fgo::constants::nanosec2sec * config_.offsetTimeSeconds.nanoseconds());

        RCLCPP_INFO_STREAM(*logger_,
                           "[Factors] Noise Model: " << fgo::utils::ROBUST2str(config_.noiseModel));

        RCLCPP_INFO_STREAM(*logger_, "[GMM] Number Components: " << config_.numComponents);
        RCLCPP_INFO_STREAM(*logger_, "[GMM] Base Sigma: " << config_.baseStdDev);
        RCLCPP_INFO_STREAM(*logger_, "[GMM] Wishart DOF: " << config_.wishartDOF);

        RCLCPP_INFO_STREAM(*logger_, "[Estimation] Tuning Algorithm: "
                << fgo::utils::ErrorModelTuningType2str(config_.tuningAlgorithm));
        RCLCPP_INFO_STREAM(*logger_,
                           "[Estimation] Prune Small: " << fgo::utils::bool2str(config_.removeSmallComponents));
        RCLCPP_INFO_STREAM(*logger_,
                           "[Estimation] Merge Similar: " << fgo::utils::bool2str(config_.mergeSimilarComponents));
        RCLCPP_INFO_STREAM(*logger_, "[Estimation] Merge Threshold: " << config_.mergingThreshold);
        RCLCPP_INFO_STREAM(*logger_,
                           "[Estimation] Estimate Mean: " << fgo::utils::bool2str(config_.estimateMean));
        RCLCPP_INFO_STREAM(*logger_, "[Estimation] Max Iterations: " << config_.maxIterations);
        RCLCPP_INFO_STREAM(*logger_, "[Estimation] Min Change: " << config_.minLikelihoodChange);
        RCLCPP_INFO_STREAM(*logger_, "[Estimation] Remove Offset: " << fgo::utils::bool2str(config_.removeOffset));

        RCLCPP_INFO_STREAM(*logger_,
                           "[ROS] Publish Mean Shifted GMM: " << fgo::utils::bool2str(config_.publishMeanShifted));
    }

    void InferenceEngine::initTypeNamesMap() {
        typeNamesMapPtr_ = makeSharedUnorderedMap<std::type_index, FACTOR>({
                                                                                   std::make_pair(std::type_index(
                                                                                                          typeid(fgo::factors::DDCarrierPhaseFactor)),
                                                                                                  FACTOR::DDCP),
                                                                                   std::make_pair(std::type_index(
                                                                                                          typeid(fgo::factors::DDPRDRFactor)),
                                                                                                  FACTOR::DDPRDR),
                                                                                   std::make_pair(std::type_index(
                                                                                                          typeid(fgo::factors::GPInterpolatedDDCarPhaFactorO)),
                                                                                                  FACTOR::GPDDCP),
                                                                                   std::make_pair(std::type_index(
                                                                                                          typeid(fgo::factors::GPInterpolatedDDPseuRaFactorO)),
                                                                                                  FACTOR::GPDDPR),
                                                                                   std::make_pair(std::type_index(
                                                                                                          typeid(fgo::factors::GPInterpolatedPRDRFactorO)),
                                                                                                  FACTOR::GPPRDR),
                                                                                   std::make_pair(std::type_index(
                                                                                                          typeid(fgo::factors::GPInterTDCPFactorO)),
                                                                                                  FACTOR::GPTDCP),
                                                                                   std::make_pair(std::type_index(
                                                                                                          typeid(fgo::factors::GPInterpolated3TDCarPhaFactorO)),
                                                                                                  FACTOR::GP3TDCP),
                                                                                   std::make_pair(std::type_index(
                                                                                                          typeid(fgo::factors::PRDRFactor)),
                                                                                                  FACTOR::PRDR),
                                                                                   std::make_pair(std::type_index(
                                                                                                          typeid(fgo::factors::PRFactor)),
                                                                                                  FACTOR::PR),
                                                                                   std::make_pair(std::type_index(
                                                                                                          typeid(fgo::factors::PRFactorGMM)),
                                                                                                  FACTOR::PRGMM),
                                                                                   std::make_pair(std::type_index(
                                                                                                          typeid(fgo::factors::PRFactorMEst)),
                                                                                                  FACTOR::PRMEST),
                                                                                   std::make_pair(std::type_index(
                                                                                                          typeid(fgo::factors::DRFactor)),
                                                                                                  FACTOR::DR),
                                                                                   std::make_pair(std::type_index(
                                                                                                          typeid(fgo::factors::TripleDiffCPFactor)),
                                                                                                  FACTOR::TDCP),
                                                                                   std::make_pair(std::type_index(
                                                                                                          typeid(fgo::factors::AmbiguityLockFactor)),
                                                                                                  FACTOR::TDCPLOCK),
                                                                                   std::make_pair(std::type_index(
                                                                                                          typeid(fgo::factors::Pseudorange3)),
                                                                                                  FACTOR::PR3)
                                                                           });
    }

    void InferenceEngine::initTopicNames() {
        topicNamesPtr_ = std::make_shared<std::vector<std::string>>(std::initializer_list<std::string>{
                "ResidualDDCP",
                "ResidualDDPRDR",
                "ResidualGPDDCP",
                "ResidualGPDDPRDR",
                "ResidualGPDDPR",
                "ResidualGPPRDR",
                "ResidualGPTDCP",
                "ResidualGP3TDCP",
                "ResidualPRDR",
                "ResidualPR",
                "ResidualPRGMM",
                "ResidualPRMEST",
                "ResidualDR",
                "ResidualTDCP",
                "ResidualTDCPLOCK",
                "ResidualPR3"
        });
    }

}