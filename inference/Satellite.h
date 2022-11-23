//
// Created by felix on 12.06.22.
//

#ifndef ONLINE_FGO_SATELLITE_H
#define ONLINE_FGO_SATELLITE_H

#include <typeinfo>
#include <typeindex>
#include <unordered_map>
#include <exception>
#include <queue>
#include <utility>
#include <algorithm>
#include <iterator>
#include <vector>
#include <array>

#include <rclcpp/rclcpp.hpp>
#include <boost/circular_buffer.hpp>
#include <gtsam/linear/NoiseModel.h>
#include <libRSF/error_models/GaussianMixture.h>
#include <libRSF/TimeMeasurement.h>
#include <libmix4sam/robust/NoiseModelNew.h>

#include <geometry_msgs/msg/vector3.hpp>
#include <irt_msgs/msg/gaussian_mixture.hpp>
#include <irt_msgs/msg/gaussian_component.hpp>
#include <irt_msgs/msg/residual.hpp>
#include <irt_msgs/msg/satellite_data.hpp>
#include <irt_msgs/msg/pr_factor_weights.hpp>
#include <ublox_msgs/msg/nav_clock.hpp>

#include "factors/GenericTypes.h"
#include "factors/measurement/gnss/DDCPFactor.h"
#include "factors/measurement/gnss/DDPRDRFactor.h"
#include "factors/measurement/gnss/GPInterDDCPFactorO.h"
#include "factors/measurement/gnss/GPInterDDPRDRFactorO.h"
#include "factors/measurement/gnss/GPInterPRDRFactorO.h"
#include "factors/measurement/gnss/GPInterTDCarPhaFactorO.h"
#include "factors/measurement/gnss/PRDRFactorO.h"
#include "include/factors/measurement/gnss/unused/TDCPFactor.h"
#include "factors/measurement/gnss/PRFactorO.h"
#include "factors/measurement/gnss/DRFactorO.h"
#include "factors/measurement/gnss/PRFactorWeighted.h"
#include "factors/smartloc/Pseudorange3.h"
#include "inference/NoiseModel.h"

#include "graphs/GraphBase.h"
#include "utils/ParamUtils.h"
#include "utils/StatUtils.h"
#include "inference/GaussianMixture.h"

namespace fgo::inference {
    using Residual = fgo::data_types::Residual;
    using ResidualsBase = fgo::data_types::ResidualsBase;
    template<int Dim>
    using Residuals = fgo::data_types::Residuals<Dim>;
    using InferenceConfig = fgo::data_types::InferenceConfig;
    using FACTOR = fgo::data_types::FACTOR;
    using ERRORCODE = fgo::data_types::ERRORCODE;
    using ROBUST = fgo::data_types::ROBUST;
    using ENVIRONMENT = fgo::data_types::ENVIRONMENT;
    using TypeNamesMapPtr = std::shared_ptr<std::unordered_map<std::type_index, FACTOR>>;
    using TopicNamesPtr = std::shared_ptr<std::vector<std::string>>;
    using idxQueuePtr = std::shared_ptr<std::queue<size_t>>;
    using ResidualsArray = std::array<std::shared_ptr<ResidualsBase>, FACTOR::END>;
    using IdxArray = std::array<std::shared_ptr<std::queue<size_t>>, FACTOR::END>;
    using ResidualPublisherPtr = rclcpp::Publisher<irt_msgs::msg::Residual>::SharedPtr;
    using MixturePublisherPtr = rclcpp::Publisher<irt_msgs::msg::GaussianMixture>::SharedPtr;
    using SatelliteDataPublisherPtr = rclcpp::Publisher<irt_msgs::msg::SatelliteData>::SharedPtr;
    using PRWeightsPublisherPtr = rclcpp::Publisher<irt_msgs::msg::PRFactorWeights>::SharedPtr;

    using namespace std::chrono_literals;

    template<typename T>
    Eigen::Matrix<T, 1, 1, Eigen::DontAlign>
    inline makeConstMatrix(const T &value) {
        Eigen::Matrix<T, 1, 1, Eigen::DontAlign> constMatrix;
        constMatrix << value;
        return constMatrix;
    }

    template<typename T, int Rows, int Cols>
    Eigen::Matrix<T, Rows, Cols, Eigen::DontAlign>
    inline makeConstMatrix(const std::vector<T> &values) {
        if (Rows * Cols != values.size()) throw std::runtime_error("Matrix size and number of elements dont match");
        Eigen::Matrix<T, Rows, Cols, Eigen::DontAlign> constMatrix;
        std::for_each(values.cbegin(), values.cend(), [&](const T &value) -> void {
            constMatrix << value;
        });
        return constMatrix;
    }

    class Satellite {

    public:

        Satellite(size_t idSat, InferenceConfig config, TopicNamesPtr topicNames,
                  const rclcpp::Node::SharedPtr &nodePtr, const rclcpp::Clock::SharedPtr &systemClockPtr);
        ~Satellite() = default;

        [[nodiscard]] idxQueuePtr getIndexQueue(FACTOR factorType) const;

        [[nodiscard]] std::vector<FACTOR> getRegisteredFactorTypes() const;

        bool addResidual(FACTOR factorType, const rclcpp::Time &timestamp, const gtsam::Vector &vec);

        std::pair<ERRORCODE, double> runInference(FACTOR factorType);

        libmix4sam::Mixture getMixtureModel(FACTOR factorType);

        template <int Dim>
        fgo::inference::GaussianMixture<Dim> getRawMixtureModel(FACTOR factorType);

        gtsam::SharedNoiseModel getNoiseModel(FACTOR factorType);

        gtsam::SharedNoiseModel getNoiseModel(FACTOR factorType, ROBUST noiseModelType);

        void setMixtureModel(FACTOR factorType, const libmix4sam::Mixture &mixtureModel);

        void updateFactorIndex(FACTOR factorType, size_t factorIndex);

        irt_msgs::msg::GaussianMixture buildMixtureMessage(FACTOR factorType, double optRuntime);

        [[nodiscard]] irt_msgs::msg::Residual buildResidualMessage(FACTOR factorType, const rclcpp::Time &timestamp, const gtsam::Vector &resVec, bool isOutlier) const;

        bool isValidNoiseModel(FACTOR factorType);

    private:

        size_t idSat_; // satellite ID
        InferenceConfig config_;
        TopicNamesPtr topicNames_; // map factor type enum -> publisher topic name
        fgo::utils::Timer timer_;
        ResidualsArray residualsArray_;
        IdxArray idxArray_;
        std::vector<FACTOR> registeredFactorTypes_;
        std::mutex mutex_;
        std::mutex cacheMutex_;
        rclcpp::Node::SharedPtr nodePtr_;
        rclcpp::Clock::SharedPtr systemClockPtr_;
        rclcpp::Time lastResidualTimestamp_;
        fgo::utils::Timer lastResidualTimer_;

        gtsam::Matrix getResidualMatrix(FACTOR factorType);

        gtsam::Matrix getFullResidualMatrix(FACTOR factorType);

        template<int Dim>
        std::pair<ERRORCODE, double> runInferenceNaive_(FACTOR factorType);

        template<int Dim>
        std::pair<ERRORCODE, double> runInferenceLongTerm_(FACTOR factorType);

        template<int Dim>
        double updateMixtureModel(fgo::inference::GaussianMixture<Dim> &gmm, const gtsam::Matrix &data);

        template<int Dim>
        irt_msgs::msg::GaussianMixture buildMixtureMessage_(FACTOR factorType, double optRuntime);

        template<int Dim>
        libmix4sam::Mixture getMixtureModel_(FACTOR factorType);

        template<int Dim>
        void setMixtureModel_(FACTOR factorType, const libmix4sam::Mixture &mixtureModel);

        template<int Dim>
        typename fgo::inference::GaussianMixture<Dim>::EstimationConfig initEstimationConfig();

        static size_t getResidualDimension(FACTOR factorType);
    };

    template<int Dim>
    double Satellite::updateMixtureModel(GaussianMixture<Dim> &gmm, const gtsam::Matrix &data) {
        // Implements Complexity Learning from
        // Incrementally learned Mixture Models for GNSS Localization (2019), Pfeifer, T., Protzel, P.
        const auto config = initEstimationConfig<Dim>();

        timer_.reset();
        if (gmm.getNumberOfComponents() == 0)
            gmm.initSpread(1, config_.baseStdDev);
        const gtsam::Matrix covStatistics = libRSF::EstimateSampleCovariance<Dim>(-data);
        if (static_cast<size_t>(gmm.getNumberOfComponents()) >= config_.numComponents) {
            gmm.sortComponentsByWeight();
            gmm.removeLastComponent();
        }
        libRSF::GaussianComponent<Dim> Component;
        Component.setParamsCovariance(covStatistics,
                                      libRSF::VectorStatic<Dim>::Zero(),
                                      gtsam::Vector1::Ones() / gmm.getNumberOfComponents());
        gmm.addComponent(Component);

        gmm.estimateOL(-data, config); // No idea why factor -1.0, consistent with libRSF behaviour tho
        return timer_.getMilliseconds();
    }

    template<int Dim>
    std::pair<ERRORCODE, double> Satellite::runInferenceNaive_(FACTOR factorType) {
        // Retrieves Residuals and downcasts it to derived type holding the gmm<Dim>
        const auto residualsBasePtr = residualsArray_[factorType];
        if (lastResidualTimer_.getDuration() < config_.waitTimeBetweenRuns)
            return std::make_pair(ERRORCODE::WAIT, .0);
        lastResidualTimer_.reset();
        // Factor somehow registered but no Residuals object constructed
        if (!residualsBasePtr)
            return std::make_pair(ERRORCODE::EMPTY, .0);
        if (!residualsBasePtr->hasNewData)
            return std::make_pair(ERRORCODE::DATA, .0);
        const auto residualsPtr = std::dynamic_pointer_cast<Residuals<Dim>>(residualsBasePtr);

        // Get the data points that represent ground truth noise distribution
        gtsam::Matrix residualMatrix;
        // All data points in time window specified by config
        residualMatrix = getResidualMatrix(factorType);
        // Empty residual matrix means no sufficient residual information available for performing inference
        if (!residualMatrix.size())
            return std::make_pair(ERRORCODE::DATA, .0);

        residualsPtr->residualsCached = residualMatrix;

        const double elapsedTime = updateMixtureModel<Dim>(residualsPtr->gmm, residualMatrix);
        residualsPtr->timer.reset();
        std::unique_lock<std::mutex> lk(cacheMutex_);
        // Update the cached GMM
        residualsPtr->gmmCached = residualsPtr->gmm;
        // We correct the offset only in cached version of the GMM
        // This allows for publishing of both GMM versions
        // Potentially also has some convergence benefits since we can initialize
        // our next inference run with the true, unshifted GMM
        if (config_.removeOffset)
            residualsPtr->gmmCached.removeOffsetOL();
        //residualsPtr->gmmCached.printParameters(nodePtr_);
        lk.unlock();
        residualsPtr->timeLastInference = systemClockPtr_->now();
        // We only allow the usage of trained GMMs as noise model for FGO
        if (!residualsPtr->isInitialized)
            residualsPtr->isInitialized = true;
        residualsBasePtr->hasNewData = false;
        return std::make_pair(ERRORCODE::SUCCESS, elapsedTime);
    }

    template<int Dim>
    std::pair<ERRORCODE, double> Satellite::runInferenceLongTerm_(FACTOR factorType) {
        // Retrieves Residuals and downcasts it to derived type holding the gmm<Dim>
        const auto residualsBasePtr = residualsArray_[factorType];
        // Factor somehow registered but no Residuals object constructed
        if (!residualsBasePtr)
            return std::make_pair(ERRORCODE::EMPTY, .0);
        const auto residualsPtr = std::dynamic_pointer_cast<Residuals<Dim>>(residualsBasePtr);
        // Not enough time passed since last inference for this factor
        if (residualsPtr->timer.getDuration() < config_.waitTimeBetweenRuns)
            return std::make_pair(ERRORCODE::WAIT, .0);
        // Get the data points that represent ground truth noise distribution
        gtsam::Matrix residualMatrix;
        // All data points in time window specified by config
        residualMatrix = getFullResidualMatrix(factorType);
        // Empty residual matrix means no sufficient residual information available for performing inference
        if (!residualMatrix.size())
            return std::make_pair(ERRORCODE::DATA, .0);

        residualsPtr->residualsCached = residualMatrix;

        const double elapsedTime = updateMixtureModel<Dim>(residualsPtr->gmm, residualMatrix);
        residualsPtr->timer.reset();
        std::unique_lock<std::mutex> lk(cacheMutex_);
        // Update the cached GMM
        residualsPtr->gmmCached = residualsPtr->gmm;
        // We correct the offset only in cached version of the GMM
        // This allows for publishing of both GMM versions
        // Potentially also has some convergence benefits since we can initialize
        // our next inference run with the true, unshifted GMM
        if (config_.removeOffset)
            residualsPtr->gmmCached.removeOffsetOL();
        //residualsPtr->gmmCached.printParameters(nodePtr_);
        lk.unlock();
        residualsPtr->timeLastInference = systemClockPtr_->now();
        // We only allow the usage of trained GMMs as noise model for FGO
        if (!residualsPtr->isInitialized)
            residualsPtr->isInitialized = true;
        return std::make_pair(ERRORCODE::SUCCESS, elapsedTime);
    }

    template<int Dim>
    irt_msgs::msg::GaussianMixture Satellite::buildMixtureMessage_(FACTOR factorType, double optRuntime) {
        irt_msgs::msg::GaussianMixture msg;
        msg.header.stamp = systemClockPtr_->now();
        msg.sat = idSat_;
        msg.factor = topicNames_->at(factorType);
        msg.environment = "None";
        msg.runtime = optRuntime;
        const auto residualsPtr = std::dynamic_pointer_cast<Residuals<Dim>>(residualsArray_[factorType]);
        msg.dim = Dim;
        Eigen::Map<gtsam::Vector> flattenedResiduals(residualsPtr->residualsCached.data(),
                                                     residualsPtr->residualsCached.size());
        msg.residuals.resize(flattenedResiduals.size());
        Eigen::Map<gtsam::Vector>(msg.residuals.data(), static_cast<long>(msg.residuals.size())) = flattenedResiduals;
        std::vector<libRSF::GaussianComponent<Dim>> mixComponents;
        std::unique_lock<std::mutex> lk(cacheMutex_);
        if (config_.publishMeanShifted)
            residualsPtr->gmmCached.getMixture(mixComponents);
        else
            residualsPtr->gmm.getMixture(mixComponents);
        lk.unlock();
        std::for_each(mixComponents.cbegin(), mixComponents.cend(),
                      [&](libRSF::GaussianComponent<Dim> mixComponent) -> void {
                          irt_msgs::msg::GaussianComponent gaussianComponent;
                          gaussianComponent.weight = mixComponent.getWeight().value();
                          const auto meanVec = mixComponent.getMean();
                          gaussianComponent.mean.resize(meanVec.size());
                          Eigen::Map<gtsam::Vector>(gaussianComponent.mean.data(),
                                                    static_cast<long>(gaussianComponent.mean.size())) = meanVec;
                          auto covMat = mixComponent.getCovariance();
                          const Eigen::Map<gtsam::Vector> covVec(covMat.data(), covMat.size());
                          gaussianComponent.covariance.resize(covMat.size());
                          Eigen::Map<gtsam::Vector>(gaussianComponent.covariance.data(),
                                                    static_cast<long>(gaussianComponent.covariance.size())) = covVec;
                          msg.gaussian.push_back(gaussianComponent);
                      });
        return msg;
    }

    template<int Dim>
    libmix4sam::Mixture Satellite::getMixtureModel_(FACTOR factorType) {
        libmix4sam::Mixture gaussianMixture;
        // Retrieves residuals and downcasts it to derived type holding the gmm<Dim>
        const auto residualsPtr = std::dynamic_pointer_cast<Residuals<Dim>>(residualsArray_[factorType]);
        std::vector<libRSF::GaussianComponent<Dim>> mixComponents;
        std::unique_lock<std::mutex> lk(cacheMutex_);
        residualsPtr->gmmCached.getMixture(mixComponents);
        lk.unlock();
        std::for_each(mixComponents.cbegin(), mixComponents.cend(), [&](libRSF::GaussianComponent<Dim> mixComponent) {
            // Transformation libRSF component -> libmix4sam component (that holds the gtsam noise model)
            gaussianMixture.add(libmix4sam::MixComponent(
                    gtsam::noiseModel::Gaussian::SqrtInformation(mixComponent.getSqrtInformation()),
                    mixComponent.getWeight().value(),
                    mixComponent.getMean()));
        });
        return gaussianMixture;
    }

    template<int Dim>
    fgo::inference::GaussianMixture<Dim> Satellite::getRawMixtureModel(FACTOR factorType) {
        libmix4sam::Mixture gaussianMixture;
        // Retrieves residuals and downcasts it to derived type holding the gmm<Dim>
        const auto residualsPtr = std::dynamic_pointer_cast<Residuals<Dim>>(residualsArray_[factorType]);
        std::unique_lock<std::mutex> lk(cacheMutex_);
        const fgo::inference::GaussianMixture<Dim> rawMixtureModel = residualsPtr->gmmCached;
        return rawMixtureModel;
    }

    template<int Dim>
    void Satellite::setMixtureModel_(FACTOR factorType, const libmix4sam::Mixture &mixtureModel) {
        fgo::inference::GaussianMixture<Dim> gaussianMixture;
        // Retrieves residuals and downcasts it to derived type holding the gmm<Dim>
        const auto residualsPtr = std::dynamic_pointer_cast<Residuals<Dim>>(residualsArray_[factorType]);
        std::for_each(mixtureModel.cbegin(), mixtureModel.cend(),
                      [&](const libmix4sam::MixComponent &mixComponent) -> void {
                          // Transformation libmix4sam component -> libRSF component
                          libRSF::GaussianComponent<Dim> gaussianComponent;
                          gaussianComponent.setParamsStdDev(
                                  mixComponent.noiseModel()->sigmas().diagonal(),
                                  mixComponent.mu(),
                                  makeConstMatrix(mixComponent.w()));
                          gaussianMixture.addComponent(gaussianComponent);
                      });

        residualsPtr->gmm = gaussianMixture;
    }

    template<int Dim>
    typename fgo::inference::GaussianMixture<Dim>::EstimationConfig Satellite::initEstimationConfig() {
        typename fgo::inference::GaussianMixture<Dim>::EstimationConfig estimationConfig;
        estimationConfig.EstimationAlgorithm = config_.tuningAlgorithm;
        estimationConfig.EstimateMean = config_.estimateMean;
        estimationConfig.MaxIterations = config_.maxIterations;
        estimationConfig.MinLikelihoodChange = config_.minLikelihoodChange;
        estimationConfig.MinimalSamples = config_.minSamples;
        //estimationConfig.PriorDirichletConcentration = config_.priorDirichletConcentration;
        //estimationConfig.PriorNormalInfoScaling = config_.priorNormalInfoScaling;
        //estimationConfig.PriorNormalMean = config_.priorNormalMean;
        estimationConfig.PriorWishartDOF = config_.wishartDOF;
        //estimationConfig.PriorWishartScatter = config_.priorWishartScatter;
        estimationConfig.RemoveSmallComponents = config_.removeSmallComponents;
        //estimationConfig.MinSamplePerComponent = config_.minSamplePerComponent;
        estimationConfig.MergeSimilarComponents = config_.mergeSimilarComponents;
        estimationConfig.MergingThreshold = config_.mergingThreshold;
        return estimationConfig;
    }
}

#endif //ONLINE_FGO_SATELLITE_H