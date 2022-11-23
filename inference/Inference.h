//
// Created by felix on 23.05.22.
//

#ifndef ONLINE_FGO_INFERENCE_H
#define ONLINE_FGO_INFERENCE_H

#include "graphs/GraphBase.h"
#include "inference/Satellite.h"

namespace fgo::inference {

    template<typename Key, typename Value>
    std::shared_ptr<std::unordered_map<Key, Value>>
    inline makeSharedUnorderedMap(
            std::initializer_list<typename std::unordered_map<Key, Value>::value_type> initList) {
        return std::make_shared<std::unordered_map<Key, Value>>(initList);
    }

    class InferenceEngine {
    protected:

        using This = InferenceEngine;

    public:

        using shared_ptr = std::shared_ptr<This>;
        using unique_ptr = std::unique_ptr<This>;

        /**
         * @brief Constructor for the Inference Engine class
         * @param config InferenceConfig that holds parameterization of the InferenceEngine
         */
        explicit InferenceEngine(InferenceConfig config);

        ~InferenceEngine() = default;

        /**
         * @brief Runs the Inference Engine for each factor that is registered with each satellite if preconditions for
         *        inference are met (sufficient amount of residual data, sufficient wait time since last run, ...)
         */
        void run();

        /**
         * @brief Multi-threaded implementation of run
         */
        void threadedRun();

        /**
         * @brief Updates the residual data of stored factors for all recognized satellites
         * @param graph Constructed factor graph for the current optimization step
         * @param result Estimated state values for the current optimization step
         * @param timestamp Timestamp corresponding to the current optimization step
         */
        void updateResidualData(const fgo::graphs::GraphBase &graph, const gtsam::Values &result,
                                const rclcpp::Time &timestamp);

        /**
         * @brief Registers index of factor in factor graph associated with a satellite observation
         * @tparam FactorType Type of factor derived from MixtureModelFactor
         * @param satId Id of the observation's satellite
         * @param factorIndex Index denoting the position of the factor of type factorType in the factor graph
         */
        template<typename FactorType>
        void registerFactor(uint32_t satId, size_t factorIndex);

        /**
         * @brief Retrieves GMM holding the gtsam noise model for factor of factorType associated with satellite
         * @tparam FactorType Type of factor derived from MixtureModelFactor
         * @param satId Id of the observation's satellite
         * @return GMM holding the gtsam noise model
         */
        template<typename FactorType>
        [[maybe_unused]] libmix4sam::Mixture getMixtureModel(uint32_t satId);

        /**
         * @brief Retrieves the raw GMM with access to its components information matrices, means and weights
         * @tparam FactorType Type of factor derived from MixtureModelFactor#
         * @tparam Dim Dimensionality of the underlying GMM
         * @param satId Id of the observation's satellite
         * @return raw GMM
         */
        template<typename FactorType, int Dim>
        [[maybe_unused]] std::vector<libRSF::GaussianComponent<Dim>> getRawMixtureModel(uint32_t satId);

        /**
         * @brief Retrieves trained gtsam noise model of type stored in config for factor of factorType associated with satellite
         * @tparam FactorType Type of factor derived from MixtureModelFactor
         * @param satId Id of the observation's satellite
         * @return shared pointer to the gtsam noise model base class
         */
        template<typename FactorType>
        gtsam::SharedNoiseModel getNoiseModel(uint32_t satId);

        /**
         * @brief Retrieves trained gtsam noise model of noiseModelType for factor of factorType associated with satellite
         * @tparam FactorType Type of factor derived from MixtureModelFactor
         * @param satId Id of the observation's satellite
         * @param noiseModelType type of the noise model that will be allocated on the heap
         * @return shared pointer to the gtsam noise model base class pointing to noise model of type noiseModelType
         */
        template<typename FactorType>
        gtsam::SharedNoiseModel getNoiseModel(uint32_t satId, ROBUST noiseModelType);

        /**
         * @brief Updates the stored gtsam noise model for factor of factorType associated with satellite
         * @tparam FactorType Type of factor derived from MixtureModelFactor
         * @param satId Id of the observation's satellite
         * @param mixtureModel GMM holding the gtsam noise model
         */
        template<typename FactorType>
        void setMixtureModel(uint32_t satId, const libmix4sam::Mixture &mixtureModel);

        /**
         * @brief Gets the FACTOR enum value corresponding to FactorType
         * @tparam FactorType Type of factor derived from MixtureModelFactor
         * @return FACTOR enum value
         */
        template<typename FactorType>
        [[nodiscard]] FACTOR getFactorEnum() const;

        /**
         * @brief Passes pointer to ROS node that Inference Engine is instantiated in and (potentially) inits publishers
         * @param nodePtr shared pointer to ROS node of constructor's caller
         * @param initPublishers flag that determines if residual and gmm data will be published
         */
        void setNode(const rclcpp::Node::SharedPtr &nodePtr, bool initPublishers = true);

        /**
         * Publishes GNSSVariance message
         * @tparam FactorType Type of factor derived from MixtureModelFactor
         * @param satId Id of the observation's satellite
         * @param gnssVar Variance associated with GNSS observation of PR
         */
        template<typename FactorType>
        void publishSatelliteData(const data_types::GNSSObs &obs);

        /**
         * Publishes PRFactorWeights message containing weights of PRFactorWeighted implementation and some statistics
         * @param satIdWeightsMap Map of satellite ids to corresponding weights
         */
        void publishPRFactorWeights(const std::map<uint32_t, gtsam::Vector1> &satIdWeightsMap);

        /**
         * Publishes PRFactorWeights message containing weights of PRFactorMultiHypothesis implementation and some statistics
         * @param satIdWeightsMap Map of satellite ids to corresponding weights
         */
        void publishPRFactorWeights(const std::map<uint32_t, fgo::data_types::DiscreteValue> &satIdWeightsMap);

        /**
         * @brief Returns truth value if noise model for factor type of satellite id is up-to-date and initialized
         * @tparam FactorType Type of factor derived from MixtureModelFactor
         * @param satId Id of the observation's satellite
         */
        template<typename FactorType>
        bool isValidNoiseModel(uint32_t satId);

    private:

        InferenceConfig config_;
        std::mutex rosMutex_;
        std::array<std::shared_ptr<Satellite>, fgo::constants::maxSatellites> satellitesArray_; // holds the Satellite objects that are responsible for residual data and gmm
        std::vector<size_t> registeredSatelliteIds_; // holds the encountered satellite ids since construction
        size_t numRegisteredSatellitesLastRun_;
        std::array<std::unique_ptr<std::thread>, fgo::constants::maxSatellites> inferenceThreads_;
        std::array<std::mutex, fgo::constants::maxSatellites> inferenceMutexes_;
        std::array<std::condition_variable, fgo::constants::maxSatellites> inferenceCVs_;
        std::array<bool, fgo::constants::maxSatellites> inferenceTriggers_;
        TypeNamesMapPtr typeNamesMapPtr_; // maps typeid of factors to enum representation
        TopicNamesPtr topicNamesPtr_; // holds string representations of factors
        rclcpp::Node::SharedPtr nodePtr_;
        std::unique_ptr<rclcpp::Logger> logger_;
        ResidualPublisherPtr resPublisher_;
        MixturePublisherPtr gmmPublisher_;
        SatelliteDataPublisherPtr satDataPublisher_;
        PRWeightsPublisherPtr prFactorWeightsPublisher_;
        std::shared_ptr<rclcpp::Clock> systemClockPtr_;

        template<typename DerivedFactor>
        gtsam::Vector computeResidual(const fgo::graphs::GraphBase &graph, const gtsam::Values &result, size_t idx);

        void printConfig();

        void initTypeNamesMap();

        void initTopicNames();

    };

    template<typename FactorType>
    void InferenceEngine::registerFactor(uint32_t satId, size_t factorIndex) {
        auto &satellite = satellitesArray_[satId];
        // If we encounter new Satellite ID, emplace it in map (saves copy-construction)
        if (!satellite) {
            satellite = std::make_shared<Satellite>(satId, config_, topicNamesPtr_, nodePtr_, systemClockPtr_);
            satellitesArray_[satId] = satellite;
            registeredSatelliteIds_.push_back(satId);
        }
        const auto factorType = typeNamesMapPtr_->at(std::type_index(typeid(FactorType)));
        // Put the index in the queue
        satellite->updateFactorIndex(factorType, factorIndex);
    }

    template<typename FactorType>
    [[maybe_unused]] libmix4sam::Mixture InferenceEngine::getMixtureModel(uint32_t satId) {
        const auto &satellite = satellitesArray_[satId];
        // We never want to return untrained GMM
        if (!satellite)
            throw std::runtime_error("Trying to get mixture model for unregistered factor. Should not happen...");
        const auto factorType = typeNamesMapPtr_->at(std::type_index(typeid(FactorType)));
        return satellite->getMixtureModel(factorType);
    }

    template<typename FactorType, int Dim>
    [[maybe_unused]] std::vector<libRSF::GaussianComponent<Dim>> InferenceEngine::getRawMixtureModel(uint32_t satId) {
        const auto &satellite = satellitesArray_[satId];
        // We never want to return untrained GMM
        if (!satellite)
            throw std::runtime_error("Trying to get mixture model for unregistered factor. Should not happen...");
        const auto factorType = typeNamesMapPtr_->at(std::type_index(typeid(FactorType)));
        const fgo::inference::GaussianMixture<Dim> &mixtureModel = satellite->getRawMixtureModel<Dim>(factorType);
        return mixtureModel.getRawMixture();
    }

    template<typename FactorType>
    [[maybe_unused]] gtsam::SharedNoiseModel InferenceEngine::getNoiseModel(uint32_t satId) {
        const auto it = std::find(registeredSatelliteIds_.cbegin(), registeredSatelliteIds_.cend(), satId);
        if (it == registeredSatelliteIds_.end())
            return {};
        const auto &satellite = satellitesArray_[satId];
        // We never want to return untrained noise model
        if (!satellite)
            throw std::runtime_error("Trying to get noise model for unregistered factor. Should not happen...");
        const auto factorType = typeNamesMapPtr_->at(std::type_index(typeid(FactorType)));
        return satellite->getNoiseModel(factorType);
    }

    template<typename FactorType>
    gtsam::SharedNoiseModel InferenceEngine::getNoiseModel(uint32_t satId, ROBUST noiseModelType) {
        const auto it = std::find(registeredSatelliteIds_.cbegin(), registeredSatelliteIds_.cend(), satId);
        if (it == registeredSatelliteIds_.end())
            return {};
        const auto &satellite = satellitesArray_[satId];
        // We never want to return untrained noise model
        if (!satellite)
            throw std::runtime_error("Trying to get noise model for unregistered factor. Should not happen...");
        const auto factorType = typeNamesMapPtr_->at(std::type_index(typeid(FactorType)));
        return satellite->getNoiseModel(factorType, noiseModelType);
    }

    template<typename FactorType>
    [[maybe_unused]] void InferenceEngine::setMixtureModel(uint32_t satId, const libmix4sam::Mixture &mixtureModel) {
        const auto it = std::find(registeredSatelliteIds_.cbegin(), registeredSatelliteIds_.cend(), satId);
        if (it == registeredSatelliteIds_.end())
            return;
        const auto &satellite = satellitesArray_[satId];
        // it does not make sense to set a mixture model for a factor we have not yet seen
        if (!satellite)
            return;
        const auto factorType = typeNamesMapPtr_->at(std::type_index(typeid(FactorType)));
        return satellite->setMixtureModel(factorType, mixtureModel);
    }

    template<typename FactorType>
    FACTOR InferenceEngine::getFactorEnum() const {
        return typeNamesMapPtr_->at(std::type_index(typeid(FactorType)));
    }

    template<typename FactorType>
    void InferenceEngine::publishSatelliteData(const data_types::GNSSObs &obs) {
        irt_msgs::msg::SatelliteData msg;
        const auto factorType = typeNamesMapPtr_->at(std::type_index(typeid(FactorType)));
        msg.header.frame_id = config_.name;
        msg.header.stamp = nodePtr_->now();
        msg.sat = obs.satId;
        msg.factor = topicNamesPtr_->at(factorType);
        msg.variance = obs.prVar;
        msg.elevation = obs.el;
        geometry_msgs::msg::Vector3 pos;
        pos.x = obs.satPos[0];
        pos.y = obs.satPos[1];
        pos.z = obs.satPos[2];
        msg.position = pos;
        geometry_msgs::msg::Vector3 vel;
        vel.x = obs.satVel[0];
        vel.y = obs.satVel[1];
        vel.z = obs.satVel[2];
        msg.velocity = vel;

        satDataPublisher_->publish(msg);
    }

    template<typename FactorType>
    bool InferenceEngine::isValidNoiseModel(uint32_t satId) {
        const auto it = std::find(registeredSatelliteIds_.cbegin(), registeredSatelliteIds_.cend(), satId);
        // If there is no satellite, there can be no noise model
        if (it == registeredSatelliteIds_.end())
            return false;
        const auto &satellite = satellitesArray_[satId];
        // Satellite exists, but no noise model has been trained
        if (!satellite)
            return false;
        const auto factorType = typeNamesMapPtr_->at(std::type_index(typeid(FactorType)));
        return satellite->isValidNoiseModel(factorType);
    }

    template<typename DerivedFactor>
    gtsam::Vector
    InferenceEngine::computeResidual(const graphs::GraphBase &graph, const gtsam::Values &result, size_t idx) {
        // Retrieves factor and downcasts it to factor type corresponding to index
        const auto factorPtr = boost::dynamic_pointer_cast<DerivedFactor>(graph.at(idx));
        if (!factorPtr) throw std::runtime_error("Bad cast in computeResidual. Index-Type mismatch...");
        const gtsam::KeyVector keys = factorPtr->keys();
        gtsam::Values values;

        // puts the state variables associated with factor into values
        std::for_each(keys.cbegin(), keys.cend(), [&](size_t key) -> void {
            values.insert(key, result.at(key));
        });

        /* PRGMM and PRMEST only appear together and scale the unwhitened error by psi and (1-psi) respectively
         * where psi: R -> (0, 1)
         * To recover the true pseudorange residual required for inference, we rescale the unwhitened error of
         * PRGMM with 1/psi and zero the unwhitened error of PRMEST to avoid duplicate residual data points
         */
        const auto factorType = getFactorEnum<DerivedFactor>();
        double factorRescaling;
        if (factorType == FACTOR::PRGMM) {
            const auto weight = values.at<gtsam::Vector1>(keys.back());
            factorRescaling = 1.0 / std::abs(weight.value());
        } else if (factorType == FACTOR::PRMEST)
            factorRescaling = 0.0;
        else
            factorRescaling = 1.0;


        // compute residual |h(X) - Z|
        return factorRescaling * factorPtr->unwhitenedError(values);
    }

}


#endif //ONLINE_FGO_INFERENCE_H