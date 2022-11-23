//
// Created by felix on 20.07.22.
//

#include <inference/Satellite.h>

namespace fgo::inference {


    Satellite::Satellite(size_t idSat, InferenceConfig config, TopicNamesPtr topicNames,
                         const rclcpp::Node::SharedPtr &nodePtr, const rclcpp::Clock::SharedPtr &systemClockPtr) :
            idSat_(idSat), config_(std::move(config)), topicNames_(std::move(topicNames)), mutex_(), cacheMutex_() {
        nodePtr_ = nodePtr;
        systemClockPtr_ = systemClockPtr;
        registeredFactorTypes_.reserve(FACTOR::END);
    }

    idxQueuePtr Satellite::getIndexQueue(FACTOR factorType) const {
        return idxArray_[factorType];
    }

    std::vector<FACTOR> Satellite::getRegisteredFactorTypes() const {
        return registeredFactorTypes_;
    }

    bool Satellite::addResidual(FACTOR factorType, const rclcpp::Time &timestamp, const gtsam::Vector &vec) {
        std::lock_guard<std::mutex> lg(mutex_);
        bool isOutlier = (vec.norm() > 5); //TODO: This is just a proof-of-concept, more sophisticated outlier identification algorithm will be implemented at a later time
        auto residuals = residualsArray_[factorType];
        // Construct container for residuals when first data point is registered
        if (!residuals) {
            const size_t residualDimension = getResidualDimension(factorType);
            if (residualDimension == 1) {
                residuals = std::make_shared<Residuals<1>>(config_.bufferSize);
                residualsArray_[factorType] = residuals;
            } else if (residualDimension == 2) {
                residuals = std::make_shared<Residuals<2>>(config_.bufferSize);
                residualsArray_[factorType] = residuals;
            } else
                throw std::runtime_error("Unknown factor type. This should not happen...");
        }
        if (!config_.longTerm || isOutlier) // Long-term engine only buffers outliers
            residuals->buffer.push_back(Residual(timestamp, vec));
        residuals->hasNewData = true;
        return isOutlier;
    }

    std::pair<ERRORCODE, double> Satellite::runInference(FACTOR factorType) {
        const size_t residualDimension = getResidualDimension(factorType);
        if (residualDimension == 1)
            return runInferenceNaive_<1>(factorType);
        else if (residualDimension == 2)
            return runInferenceNaive_<2>(factorType);
        else
            throw std::runtime_error("Unknown factor type. This should not happen...");
    }

    libmix4sam::Mixture Satellite::getMixtureModel(FACTOR factorType) {
        const size_t residualDimension = getResidualDimension(factorType);
        if (residualDimension == 1)
            return getMixtureModel_<1>(factorType);
        else if (residualDimension == 2)
            return getMixtureModel_<2>(factorType);
        else
            throw std::runtime_error("Unknown factor type. This should not happen...");
    }

    gtsam::SharedNoiseModel Satellite::getNoiseModel(FACTOR factorType) {
        gtsam::SharedNoiseModel noiseModel;
        // return nullptr if last inference was too long ago, no inference was performed yet, ...
        if (!isValidNoiseModel(factorType))
            return noiseModel;
        const auto mixtureModel = getMixtureModel(factorType);
        const auto noiseModelType = config_.noiseModel;
        switch (noiseModelType) {

            case ROBUST::MAXMIX:
                noiseModel = fgo::noiseModel::MaxMix::Create(mixtureModel);
                return noiseModel;
            case ROBUST::SUMMIX:
                noiseModel = libmix4sam::noiseModelNew::SumMix::Create(mixtureModel);
                return noiseModel;
            case ROBUST::MAXSUMMIX:
                noiseModel = libmix4sam::noiseModelNew::MaxSumMix::Create(mixtureModel);
                return noiseModel;
            case ROBUST::CAUCHY:
            case ROBUST::HUBER:
                throw std::runtime_error("Invalid robust noise model type");
        }
    }

    gtsam::SharedNoiseModel Satellite::getNoiseModel(FACTOR factorType, ROBUST noiseModelType) {
        gtsam::SharedNoiseModel noiseModel;
        // return nullptr if last inference was too long ago, no inference was performed yet, ...
        if (!isValidNoiseModel(factorType))
            return noiseModel;
        const auto mixtureModel = getMixtureModel(factorType);
        switch (noiseModelType) {

            case ROBUST::MAXMIX: {
                noiseModel = fgo::noiseModel::MaxMix::Create(mixtureModel);
                return noiseModel;
            }
            case ROBUST::SUMMIX: {
                noiseModel = libmix4sam::noiseModelNew::SumMix::Create(mixtureModel);
                return noiseModel;
            }
            case ROBUST::MAXSUMMIX: {
                noiseModel = libmix4sam::noiseModelNew::MaxSumMix::Create(mixtureModel);
                return noiseModel;
            }
            case ROBUST::CAUCHY:
            case ROBUST::HUBER:
                throw std::runtime_error("Invalid robust noise model type");
        }
    }

    void Satellite::setMixtureModel(FACTOR factorType, const libmix4sam::Mixture &mixtureModel) {
        const size_t residualDimension = getResidualDimension(factorType);
        if (residualDimension == 1)
            setMixtureModel_<1>(factorType, mixtureModel);
        else if (residualDimension == 2)
            setMixtureModel_<2>(factorType, mixtureModel);
        else
            throw std::runtime_error("Unknown factor type. This should not happen...");
    }

    void Satellite::updateFactorIndex(FACTOR factorType, size_t factorIndex) {
        auto &idxQueue = idxArray_[factorType];
        if (!idxQueue) {
            idxQueue = std::make_shared<std::queue<size_t>>();
            idxArray_[factorType] = idxQueue;
            registeredFactorTypes_.push_back(factorType);
        }
        idxQueue->emplace(factorIndex);
    }

    irt_msgs::msg::GaussianMixture Satellite::buildMixtureMessage(FACTOR factorType, double optRuntime) {
        const size_t residualDimension = getResidualDimension(factorType);
        if (residualDimension == 1)
            return buildMixtureMessage_<1>(factorType, optRuntime);
        else if (residualDimension == 2)
            return buildMixtureMessage_<2>(factorType, optRuntime);
        else
            throw std::runtime_error("Unknown factor type. This should not happen...");
    }

    irt_msgs::msg::Residual
    Satellite::buildResidualMessage(FACTOR factorType, const rclcpp::Time &timestamp,
                                    const gtsam::Vector &resVec, bool isOutlier) const {
        irt_msgs::msg::Residual msg;
        msg.header.stamp = timestamp;
        msg.sat = idSat_;
        msg.factor = topicNames_->at(factorType);
        msg.outlier = isOutlier;
        msg.data.resize(resVec.size());
        Eigen::Map<gtsam::Vector>(msg.data.data(), static_cast<long>(msg.data.size())) = resVec;
        return msg;
    }

    bool Satellite::isValidNoiseModel(FACTOR factorType) {
        const auto maxDurationSinceLastInference = config_.offsetTimeSeconds + config_.waitTimeBetweenRuns;
        const auto &residualsPtr = residualsArray_[factorType];
        const auto now = systemClockPtr_->now();
        if (!residualsPtr)
            return false;
        return (now - residualsPtr->timeLastInference < maxDurationSinceLastInference) && residualsPtr->isInitialized;
    }

    gtsam::Matrix Satellite::getResidualMatrix(FACTOR factorType) {
        std::lock_guard<std::mutex> lg(mutex_);
        const auto &residualsPtr = residualsArray_[factorType];
        auto &residualBuffer = residualsPtr->buffer;

        // search first residual in buffer that is within time window for valid residuals
        auto itBegin = std::upper_bound(residualBuffer.begin(), residualBuffer.end(),
                                        residualBuffer.back().timestamp - config_.windowSize,
                                        [](const rclcpp::Time &time,
                                           const Residual &res) -> bool { // binary predicate that returns true, if first arg is LESS THAN second arg
                                            return time < res.timestamp;
                                        });
        const auto numValidElements = std::distance(itBegin, residualBuffer.end());
        const auto numResidualDimensions = getResidualDimension(factorType);
        // if buffer does not hold sufficient residual information, we return empty 0x0 matrix
        if (numValidElements < static_cast<long>(config_.minSamples))
            return {};
        gtsam::Matrix mat(numResidualDimensions, numValidElements);
        long matCol = 0;
        // fill matrix with valid residuals
        // numCols == |buffer.current() - buffer.end()|
        for (; itBegin != residualBuffer.end(); ++itBegin) {
            mat.col(matCol) = itBegin->data;
            matCol++;
        }
        return mat;
    }

    gtsam::Matrix Satellite::getFullResidualMatrix(FACTOR factorType) {
        std::lock_guard<std::mutex> lg(mutex_);
        const auto &residualsPtr = residualsArray_[factorType];
        const auto &residualBuffer = residualsPtr->buffer;
        const auto numValidElements = residualBuffer.size();
        const auto numResidualDimensions = getResidualDimension(factorType);
        // if buffer does not hold sufficient residual information, we return empty 0x0 matrix
        if (numValidElements < config_.minSamples)
            return {};
        gtsam::Matrix mat(numResidualDimensions, numValidElements);
        long matCol = 0;
        // fill matrix with valid residuals
        // numCols == |buffer.current() - buffer.end()|
        for (const auto &res : residualBuffer) {
            mat.col(matCol) = res.data;
            matCol++;
        }
        return mat;
    }

    size_t Satellite::getResidualDimension(FACTOR factorType) {
        // We need the residual dimension since we need to be able to downcast to correct Residuals<Dim> at run-time.
        // Dim needs to be available at compile-time because libRSF uses it as non-type template parameter.
        // Therefore we create derived classes for all possible Dims and work with pointers to base class which
        // we downcast based on this function's return value during run-time.
        switch (factorType) {
            case FACTOR::DDCP:
            case FACTOR::GPDDCP:
            case FACTOR::GPDDPR:
            case FACTOR::GPTDCP:
            case FACTOR::GP3TDCP:
            case FACTOR::TDCP:
            case FACTOR::TDCPLOCK:
            case FACTOR::PR:
            case FACTOR::PRGMM:
            case FACTOR::PRMEST:
            case FACTOR::DR:
            case FACTOR::PR3:
                return 1;
            case FACTOR::DDPRDR:
            case FACTOR::GPDDPRDR:
            case FACTOR::GPPRDR:
            case FACTOR::PRDR:
                return 2;
            case FACTOR::END:
            default:
                return 0;
        }
    }
}