//
// Created by felix on 01.11.22.
//
#include "solvers/IncrementalFixedLagDCSAM.h"

namespace fgo::solvers {

    IncrementalFixedLagHybridSmoother::IncrementalFixedLagHybridSmoother(double smootherLag, int maxIterations,
                                                                         const gtsam::ISAM2Params &isam_params)
            : isam_params_(isam_params), smootherLag_(smootherLag), maxIterations_(maxIterations) {
        solver_ = fgo::solvers::IncrementalFixedLagSmoother(smootherLag, isam_params_);
    }

    int IncrementalFixedLagHybridSmoother::optimize(const gtsam::NonlinearFactorGraph &graph,
                                                    const gtsam::DiscreteFactorGraph &dfg, const DCFactorGraph &dcfg,
                                                    const gtsam::Values &initialGuessContinuous,
                                                    const DiscreteValues &initialGuessDiscrete,
                                                    const KeyTimestampMap &tContinuous,
                                                    const KeyTimestampMap &tDiscrete) {
        DCValues lastEstimate(initialGuessContinuous, initialGuessDiscrete);
        update(graph, dfg, dcfg, initialGuessContinuous, initialGuessDiscrete, tContinuous, tDiscrete);
        for (int i = 0; i < maxIterations_; ++i) {
            update();
            DCValues currentEstimate(currContinuous_, currDiscrete_);
            if (currentEstimate == lastEstimate)
                return i + 1;
            lastEstimate = currentEstimate;
        }
    }

    void IncrementalFixedLagHybridSmoother::update(const gtsam::NonlinearFactorGraph &graph,
                                                   const gtsam::DiscreteFactorGraph &dfg,
                                                   const DCFactorGraph &dcfg,
                                                   const gtsam::Values &initialGuessContinuous,
                                                   const DiscreteValues &initialGuessDiscrete,
                                                   const KeyTimestampMap &tContinuous,
                                                   const KeyTimestampMap &tDiscrete) {
        bool initialUpdate = (!graph.empty() || !dfg.empty() || !dcfg.empty());
        if (initialUpdate) {
            timestampsContinuous_.updateKeyTimestampMap(tContinuous);
            timestampsDiscrete_.updateKeyTimestampMap(tDiscrete);
        }

        // First things first: combine currContinuous_ estimate with the new values
        // from initialGuessContinuous to produce the full continuous variable state.
        for (const gtsam::Key k: initialGuessContinuous.keys()) {
            if (currContinuous_.exists(k))
                currContinuous_.update(k, initialGuessContinuous.at(k));
            else
                currContinuous_.insert(k, initialGuessContinuous.at(k));
        }

        // Also combine currDiscrete_ estimate with new values from
        // initialGuessDiscrete to give a full discrete variable state.
        for (const auto &kv: initialGuessDiscrete) {
            // This will update the element with key `kv.first` if one exists, or add a
            // new element with key `kv.first` if not.
            currDiscrete_[kv.first] = initialGuessDiscrete.at(kv.first);
        }

        // We'll combine the nonlinear factors with DCContinuous factors before
        // passing to the continuous solver; likewise for the discrete factors and
        // DCDiscreteFactors.
        gtsam::NonlinearFactorGraph combined;
        gtsam::DiscreteFactorGraph discreteCombined;

        // Populate combined and discreteCombined with the provided nonlinear and
        // discrete factors, respectively.
        for (auto &factor: graph) combined.add(factor);
        for (auto &factor: dfg) discreteCombined.push_back(factor);

        // Each DCFactor will be split into a separate discrete and continuous
        // component
        for (auto &dcfactor: dcfg) {
            fgo::factors::DCDiscreteFactor dcDiscreteFactor(dcfactor);
            auto sharedDiscrete =
                    boost::make_shared<fgo::factors::DCDiscreteFactor>(dcDiscreteFactor);
            discreteCombined.push_back(sharedDiscrete);
            dcDiscreteFactors_.push_back(sharedDiscrete);
        }

        if (initialUpdate)
            updateNoiseModels();

        // Set discrete information in DCDiscreteFactors.
        updateDiscrete(discreteCombined, currContinuous_, currDiscrete_);

        // Update current discrete state estimate.
        if (!initialGuessContinuous.empty() && initialGuessDiscrete.empty() &&
            discreteCombined.empty()) {
            // This is an odometry?
        } else {
            if (!initialUpdate)
                currDiscrete_ = solveDiscrete();
        }

        for (auto &dcfactor: dcfg) {
            fgo::factors::DCContinuousFactor dcContinuousFactor(dcfactor);
            auto sharedContinuous =
                    boost::make_shared<fgo::factors::DCContinuousFactor>(dcContinuousFactor);
            sharedContinuous->updateDiscrete(currDiscrete_);
            combined.push_back(sharedContinuous);
            dcContinuousFactors_.push_back(sharedContinuous);
        }

        // Only the initialGuess needs to be provided for the continuous solver (not
        // the entire continuous state).
        updateContinuousInfo(currDiscrete_, combined, initialGuessContinuous, tContinuous);
        currContinuous_ = solver_.calculateEstimate();
        // Update discrete info from last solve and
        updateDiscrete(discreteCombined, currContinuous_, currDiscrete_);
    }

    void IncrementalFixedLagHybridSmoother::update(const HybridFactorGraph &hfg,
                                                   const gtsam::Values &initialGuessContinuous,
                                                   const DiscreteValues &initialGuessDiscrete) {
        update(hfg.nonlinearGraph(), hfg.discreteGraph(), hfg.dcGraph(),
               initialGuessContinuous, initialGuessDiscrete);
    }

    void IncrementalFixedLagHybridSmoother::update() {
        update(gtsam::NonlinearFactorGraph(), gtsam::DiscreteFactorGraph(),
               DCFactorGraph());
    }

    void IncrementalFixedLagHybridSmoother::updateDiscrete(
            const gtsam::DiscreteFactorGraph &dfg = gtsam::DiscreteFactorGraph(),
            const gtsam::Values &continuousVals = gtsam::Values(),
            const DiscreteValues &discreteVals = DiscreteValues()) {
        for (auto &factor: dfg) {
            dfg_.push_back(factor);
        }
        updateDiscreteInfo(continuousVals, discreteVals);
    }

    void IncrementalFixedLagHybridSmoother::updateDiscreteInfo(const gtsam::Values &continuousVals,
                                                               const DiscreteValues &discreteVals) {
        if (continuousVals.empty()) return;
        for (auto factor: dcDiscreteFactors_) {
            boost::shared_ptr<fgo::factors::DCDiscreteFactor> dcDiscrete =
                    boost::static_pointer_cast<fgo::factors::DCDiscreteFactor>(factor);
            dcDiscrete->updateContinuous(continuousVals);
            dcDiscrete->updateDiscrete(discreteVals);
        }
    }

    void IncrementalFixedLagHybridSmoother::updateContinuousInfo(const DiscreteValues &discreteVals,
                                                                 const gtsam::NonlinearFactorGraph &newFactors,
                                                                 const gtsam::Values &initialGuess,
                                                                 const KeyTimestampMap &tContinuous) {
        solver_.update(newFactors, initialGuess, tContinuous);
        marginalizeFull();
    }

    void IncrementalFixedLagHybridSmoother::marginalizeFull() {
        marginalizeDiscrete();
        marginalizeContinuous();
    }

    void IncrementalFixedLagHybridSmoother::updateNoiseModels() {
        auto factorSet = getNonlinearFactorGraph();
        for (const auto &factor: factorSet) {
            auto dcContinuousFactor = boost::dynamic_pointer_cast<fgo::factors::DCContinuousFactor>(factor);
            if (dcContinuousFactor) {
                auto derivedFactor = boost::dynamic_pointer_cast<fgo::factors::HybridMixture<fgo::factors::Pseudorange3>>(dcContinuousFactor->factor());
                if (derivedFactor) {

                    auto &mixtureFactors = derivedFactor->getFactors();
                    if (mixtureFactors.size() == 1)
                        continue;

                    const auto satId = derivedFactor->satId();
                    auto mixtureModel = inferenceEngine_->getMixtureModel<fgo::factors::Pseudorange3>(satId);

                    auto discreteKey = derivedFactor->discreteKey();
                    dcsam::DiscreteValues activeHypothesis;
                    activeHypothesis[discreteKey.first] = 0;
                    currDiscrete_[discreteKey.first] = 0;
                    dcContinuousFactor->updateDiscrete(activeHypothesis);

                    if (mixtureFactors.size() == 2) {
                        fgo::noiseModel::MaxMix::shared_ptr noiseModel = fgo::noiseModel::MaxMix::Create(mixtureModel);
                        noiseModel->setErrorCorrection(boost::dynamic_pointer_cast<fgo::noiseModel::MaxMix>(mixtureFactors[1].noiseModel())->getErrorCorrection());
                        mixtureFactors[1].updateNoiseModel(noiseModel);
                    }
                }
            }
        }


        gtsam::ISAM2UpdateParams updateParams;
        updateParams.forceFullSolve = true;
        solver_.update(updateParams);


    }


    void IncrementalFixedLagHybridSmoother::marginalizeDiscrete() {
        const double currentTimestamp = timestampsDiscrete_.getCurrentTimestamp();
        const gtsam::KeyVector marginalizableKeys = timestampsDiscrete_.findKeysBefore(currentTimestamp - smootherLag_);

        if (marginalizableKeys.empty())
            return;

        const gtsam::KeyVector nonMarginalizableKeys = timestampsDiscrete_.findKeysAfter(currentTimestamp - smootherLag_);
        gtsam::DiscreteFactorGraph::shared_ptr marginalizedDiscreteGraph;
        gtsam::FastVector<gtsam::DiscreteFactor::shared_ptr> marginalizedDiscreteFactors;
        gtsam::DiscreteValues marginalizedDiscreteState;

        marginalizedDiscreteGraph = dfg_.marginal(nonMarginalizableKeys);

        std::for_each(dcDiscreteFactors_.begin(), dcDiscreteFactors_.end(), [&](const gtsam::DiscreteFactor::shared_ptr &factor) -> void {
            const gtsam::KeyVector factorKeys = factor->keys();
            const auto factorTimestamp = timestampsDiscrete_.getTimestamp(factorKeys.back());
            if (currentTimestamp - factorTimestamp < smootherLag_)
                marginalizedDiscreteFactors.push_back(factor);
        });

        using discreteKeyValuePairType = std::pair<const gtsam::Key, unsigned long>;
        std::for_each(currDiscrete_.begin(), currDiscrete_.end(), [&](const discreteKeyValuePairType &keyValuePair) -> void {
            const auto factorTimestamp = timestampsDiscrete_.getTimestamp(keyValuePair.first);
            if (currentTimestamp - factorTimestamp < smootherLag_)
                marginalizedDiscreteState.insert(keyValuePair);
        });

        dfg_ = *marginalizedDiscreteGraph;
        dcDiscreteFactors_ = marginalizedDiscreteFactors;
        currDiscrete_ = marginalizedDiscreteState;
        timestampsDiscrete_.eraseKeyTimestampMap(marginalizableKeys);
    }

    void IncrementalFixedLagHybridSmoother::marginalizeContinuous() {
        const double currentTimestamp = timestampsContinuous_.getCurrentTimestamp();
        const gtsam::KeyVector marginalizableKeys = timestampsContinuous_.findKeysBefore(currentTimestamp - smootherLag_);

        if (marginalizableKeys.empty())
            return;

        gtsam::NonlinearFactorGraph marginalizedContinuousGraph;
        std::vector<fgo::factors::DCContinuousFactor::shared_ptr> marginalizedContinuousFactors;
        gtsam::Values marginalizedContinuousState;

        std::for_each(fg_.begin(), fg_.end(), [&](const gtsam::NonlinearFactor::shared_ptr &factor) -> void {
            const gtsam::KeyVector factorKeys = factor->keys();
            const auto factorTimestamp = timestampsContinuous_.getTimestamp(factorKeys.back());
            if (currentTimestamp - factorTimestamp < smootherLag_)
                marginalizedContinuousGraph.push_back(factor);
        });

        std::for_each(dcContinuousFactors_.begin(), dcContinuousFactors_.end(), [&](const fgo::factors::DCContinuousFactor::shared_ptr &factor) -> void {
            const gtsam::KeyVector factorKeys = factor->keys();
            const auto factorTimestamp = timestampsContinuous_.getTimestamp(factorKeys.back());
            if (currentTimestamp - factorTimestamp < smootherLag_)
                marginalizedContinuousFactors.push_back(factor);
        });

        std::for_each(currContinuous_.begin(), currContinuous_.end(), [&](gtsam::Values::KeyValuePair keyValuePair) -> void {
            const auto factorTimestamp = timestampsContinuous_.getTimestamp(keyValuePair.key);
            if (currentTimestamp - factorTimestamp < smootherLag_)
                marginalizedContinuousState.insert(keyValuePair.key, keyValuePair.value);
        });

        fg_ = marginalizedContinuousGraph;
        dcContinuousFactors_ = marginalizedContinuousFactors;
        currContinuous_ = marginalizedContinuousState;
        timestampsContinuous_.eraseKeyTimestampMap(marginalizableKeys);
    }

    DiscreteValues IncrementalFixedLagHybridSmoother::solveDiscrete() const {
        DiscreteValues discreteVals = dfg_.optimize();
        return discreteVals;
    }

    DCValues IncrementalFixedLagHybridSmoother::calculateEstimate() const {
        // NOTE: if we have these cached from solves, we could presumably just return
        // the cached values.
        gtsam::Values continuousVals = solver_.calculateEstimate();
        DiscreteValues discreteVals = dfg_.optimize();
        DCValues dcValues(continuousVals, discreteVals);
        return dcValues;
    }

}
