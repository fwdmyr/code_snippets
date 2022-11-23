// Created by felix on 01.11.22.

#pragma once

#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "factors/hybrid/HybridMixture.h"
#include "factors/hybrid/DCContinuousFactor.h"
#include "factors/hybrid/DCDiscreteFactor.h"

#include "dcsam/HybridFactorGraph.h"

#include "factors/smartloc/Pseudorange3.h"
#include "factors/measurement/gnss/PRFactorO.h"

#include "inference/Inference.h"
#include "inference/NoiseModel.h"

#include "solvers/IncrementalFixedLagSmoother.h"

using namespace dcsam;
typedef std::map<gtsam::Key, double> KeyTimestampMap;
typedef std::multimap<double, gtsam::Key> TimestampKeyMap;

namespace fgo::solvers {

    class KeyTimestampContainer {

    public:

        KeyTimestampContainer() = default;

        ~KeyTimestampContainer() = default;

        /* ************************************************************************* */
        void updateKeyTimestampMap(const KeyTimestampMap &timestamps) {
            // Loop through each key and add/update it in the map
            for (const auto &key_timestamp: timestamps) {
                // Check to see if this key already exists in the database
                auto keyIter = keyTimestampMap_.find(key_timestamp.first);

                // If the key already exists
                if (keyIter != keyTimestampMap_.end()) {
                    // Find the entry in the Timestamp-Key database
                    std::pair<TimestampKeyMap::iterator, TimestampKeyMap::iterator> range = timestampKeyMap_.equal_range(
                            keyIter->second);
                    auto timeIter = range.first;
                    while (timeIter->second != key_timestamp.first) {
                        ++timeIter;
                    }
                    // remove the entry in the Timestamp-Key database
                    timestampKeyMap_.erase(timeIter);
                    // insert an entry at the new time
                    timestampKeyMap_.insert(TimestampKeyMap::value_type(key_timestamp.second, key_timestamp.first));
                    // update the Key-Timestamp database
                    keyIter->second = key_timestamp.second;
                } else {
                    // Add the Key-Timestamp database
                    keyTimestampMap_.insert(key_timestamp);
                    // Add the key to the Timestamp-Key database
                    timestampKeyMap_.insert(TimestampKeyMap::value_type(key_timestamp.second, key_timestamp.first));
                }
            }
        }

        /* ************************************************************************* */
        void eraseKeyTimestampMap(const gtsam::KeyVector &keys) {
            for (gtsam::Key key: keys) {
                // Erase the key from the Timestamp->Key map
                double timestamp = keyTimestampMap_.at(key);

                auto iter = timestampKeyMap_.lower_bound(timestamp);
                while (iter != timestampKeyMap_.end() && iter->first == timestamp) {
                    if (iter->second == key) {
                        timestampKeyMap_.erase(iter++);
                    } else {
                        ++iter;
                    }
                }
                // Erase the key from the Key->Timestamp map
                keyTimestampMap_.erase(key);
            }
        }

        /* ************************************************************************* */
        [[nodiscard]] double getCurrentTimestamp() const {
            if (!timestampKeyMap_.empty()) {
                return timestampKeyMap_.rbegin()->first;
            } else {
                return -std::numeric_limits<double>::max();
            }
        }

        /* ************************************************************************* */
        [[nodiscard]] gtsam::KeyVector findKeysBefore(double timestamp) const {
            gtsam::KeyVector keys;
            auto end = timestampKeyMap_.lower_bound(timestamp);
            for (auto iter = timestampKeyMap_.begin(); iter != end; ++iter) {
                keys.push_back(iter->second);

            }
            return keys;
        }

        /* ************************************************************************* */
        [[nodiscard]] gtsam::KeyVector findKeysAfter(double timestamp) const {
            gtsam::KeyVector keys;
            auto iter = timestampKeyMap_.lower_bound(timestamp);
            for (; iter != timestampKeyMap_.end(); ++iter) {
                keys.push_back(iter->second);

            }
            return keys;
        }

        /* ************************************************************************* */
        [[nodiscard]] double getTimestamp(gtsam::Key key) const {
            return keyTimestampMap_.at(key);
        }

    private:
        /** The current timestamp associated with each tracked key */
        TimestampKeyMap timestampKeyMap_;
        KeyTimestampMap keyTimestampMap_;
    };

    inline bool operator==(const dcsam::DCValues &lhs, const dcsam::DCValues &rhs) {
        return lhs.continuous.equals(rhs.continuous) && lhs.discrete.equals(rhs.discrete);
    }

    class IncrementalFixedLagHybridSmoother {
    public:
        IncrementalFixedLagHybridSmoother() = delete;

        IncrementalFixedLagHybridSmoother(double smootherLag, int maxIterations, const gtsam::ISAM2Params &isam_params);

        int optimize(const gtsam::NonlinearFactorGraph &graph,
                      const gtsam::DiscreteFactorGraph &dfg, const DCFactorGraph &dcfg,
                      const gtsam::Values &initialGuessContinuous = gtsam::Values(),
                      const DiscreteValues &initialGuessDiscrete = DiscreteValues(),
                      const KeyTimestampMap &tContinuous = KeyTimestampMap(),
                      const KeyTimestampMap &tDiscrete = KeyTimestampMap());

        /**
         * For this solver, runs an iteration of alternating minimization between
         * discrete and continuous variables, adding any user-supplied factors (with
         * initial guess) first.
         *
         * 1. Adds new discrete factors (if any) as supplied by a user to the
         * discrete factor graph, then adds any discrete-continuous factors to the
         * discrete factor graph, appropriately initializing their continuous
         * variables to those from the last solve and any supplied by the initial
         * guess.
         *
         * 2. Update the solution for the discrete variables.
         *
         * 3. For all new discrete-continuous factors to be passed to the continuous
         * solver, update/set the latest discrete variables (prior to adding).
         *
         * 4. In one step: add new factors, new values, and earmarked old factor keys
         * to iSAM. Specifically, loop over DC factors already in iSAM, updating their
         * discrete information, then call isam_.update() with the (initialized) new
         * DC factors, any new continuous factors, and the initial guess to be
         * supplied.
         *
         * 5. Calculate the latest continuous variables from iSAM.
         *
         * 6. Update the discrete factors in the discrete factor graph dfg_ with the
         * latest information from the continuous solve.
         *
         * @param graph - a gtsam::NonlinearFactorGraph containing any
         * *continuous-only* factors to add.
         * @param dfg - a gtsam::DiscreteFactorGraph containing any *discrete-only*
         * factors to add.
         * @param dcfg - a DCFactorGraph containing any joint discrete-continuous
         * factors to add.
         * @param initialGuess - an initial guess for any new continuous keys that.
         * appear in the updated factors (or if one wants to force override previously
         * obtained continuous values).
         */
        void update(const gtsam::NonlinearFactorGraph &graph,
                    const gtsam::DiscreteFactorGraph &dfg, const DCFactorGraph &dcfg,
                    const gtsam::Values &initialGuessContinuous = gtsam::Values(),
                    const DiscreteValues &initialGuessDiscrete = DiscreteValues(),
                    const KeyTimestampMap &tContinuous = KeyTimestampMap(),
                    const KeyTimestampMap &tDiscrete = KeyTimestampMap());

        /**
         * A HybridFactorGraph is a container holding a NonlinearFactorGraph, a
         * DiscreteFactorGraph, and a DCFactorGraph, so internally this function
         * simply issues a call to `update` with these internal graphs passed as
         * parameters: that is:
         *
         * update(hfg.nonlinearGraph(), hfg.discreteGraph(), hfg.dcGraph(),
         * initialGuess);
         */
        void update(const HybridFactorGraph &hfg,
                    const gtsam::Values &initialGuessContinuous = gtsam::Values(),
                    const DiscreteValues &initialGuessDiscrete = DiscreteValues());

        /**
         * Inline convenience function to allow "skipping" the initial guess for
         * continuous variables while adding an initial guess for discrete variables.
         */
        inline void update(const HybridFactorGraph &hfg,
                           const DiscreteValues &initialGuessDiscrete) {
            update(hfg, gtsam::Values(), initialGuessDiscrete);
        }

        /**
         * Inline convenience function for pure continuous update
         */
        inline void update(const gtsam::NonlinearFactorGraph &graph, const gtsam::Values &initialGuessContinuous, const KeyTimestampMap &tContinuous) {
            update(graph, gtsam::DiscreteFactorGraph(), DCFactorGraph(),
                   initialGuessContinuous, DiscreteValues(), tContinuous, KeyTimestampMap());
        }

        /**
         * Simply used to call `update` without any new factors. Runs an iteration of
         * optimization.
         */
        void update();

        /**
         * Add factors in `graph` to member discrete factor graph `dfg_`, then update
         * any stored continuous variables using those in `values` by calling
         * `updateDiscreteInfo(values)`.
         *
         * NOTE: could this be combined with `updateDiscreteInfo` or do these
         * definitely need to be separate?
         *
         * @param graph - a discrete factor graph containing the factors to add
         * @param values - an assignment to the continuous variables (or subset
         * thereof).
         */
        void updateDiscrete(const gtsam::DiscreteFactorGraph &graph,
                            const gtsam::Values &continuousVals,
                            const DiscreteValues &discreteVals);

        /**
         * For any factors in `dfg_`, update their stored local continuous information
         * with the values from `values`.
         *
         * NOTE: could this be combined with `updateDiscrete` or do these
         * definitely need to be separate?
         *
         * @param values - an assignment to the continuous variables (or subset
         * thereof).
         */
        void updateDiscreteInfo(const gtsam::Values &continuousVals,
                                const DiscreteValues &discreteVals);

        /**
         * Given the latest discrete values (dcValues), a set of new factors
         * (newFactors), and an initial guess for any new keys (initialGuess), this
         * function updates the continuous values stored in any DC factors (in the
         * member `isam_` instance), marks any affected keys as such, and calls
         * `isam_.update` with the new factors and initial guess. See implementation
         * for more detail.
         *
         * NOTE: this is another function that could perhaps be named better.
         */
        void updateContinuousInfo(const DiscreteValues &discreteVals,
                                  const gtsam::NonlinearFactorGraph &newFactors,
                                  const gtsam::Values &initialGuess,
                                  const KeyTimestampMap &tContinuous);

        void marginalizeFull();

        void marginalizeDiscrete();

        void marginalizeContinuous();

        void updateNoiseModels();

        /**
         * Solve for discrete variables given continuous variables. Internally, calls
         * `dfg_.optimize()`
         *
         * @return an assignment (DiscreteValues) to the discrete variables in the
         * graph.
         */
        DiscreteValues solveDiscrete() const;

        /**
         * This is the primary function used to extract an estimate from the solver.
         * Internally, calls `isam_.calculateEstimate()` and `dfg_.optimize()` to
         * obtain an estimate for the continuous (resp. discrete) variables and
         * packages them into a `DCValues` pair as (continuousVals, discreteVals).
         *
         * @return a DCValues object containing an estimate
         * of the most probable assignment to the continuous (DCValues.continuous) and
         * discrete (DCValues.discrete) variables.
         */
        DCValues calculateEstimate() const;

        gtsam::DiscreteFactorGraph getDiscreteFactorGraph() const { return dfg_; }

        gtsam::NonlinearFactorGraph getNonlinearFactorGraph() const {
            return solver_.getFactors();
        }

    private:
        // Global factor graph and iSAM2 instance
        gtsam::NonlinearFactorGraph fg_;  // NOTE: unused
        gtsam::ISAM2Params isam_params_;
        fgo::solvers::IncrementalFixedLagSmoother solver_;
        double smootherLag_;
        int maxIterations_;
        gtsam::DiscreteFactorGraph dfg_;
        gtsam::Values currContinuous_;
        DiscreteValues currDiscrete_;
        KeyTimestampContainer timestampsContinuous_;
        KeyTimestampContainer timestampsDiscrete_;

        std::shared_ptr<fgo::inference::InferenceEngine> inferenceEngine_;

        std::vector<fgo::factors::DCContinuousFactor::shared_ptr> dcContinuousFactors_;
        gtsam::FastVector<gtsam::DiscreteFactor::shared_ptr> dcDiscreteFactors_;
    };

}
