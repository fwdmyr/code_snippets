//
// Created by felix on 11.06.22.
//

#ifndef ONLINE_FGO_GAUSSIANMIXTURE_H
#define ONLINE_FGO_GAUSSIANMIXTURE_H

#include <libRSF/error_models/GaussianMixture.h>
#include "data/DataTypes.h"

namespace fgo::inference {

    enum class ErrorModelTuningType {None, EM, EM_MAP, VBI, VBI_Full, VBI_ParameterLearning};

    // Forward declaration
    template<int Dim>
    class GaussianMixture;

    inline void RemoveColumn(libRSF::Matrix &Matrix, int ColToRemove) {
        long numRows = Matrix.rows();
        long numCols = Matrix.cols() - 1;

        if (ColToRemove < numCols) {
            Matrix.block(0, ColToRemove, numRows, numCols - ColToRemove) = Matrix.rightCols(numCols - ColToRemove);
        }

        Matrix.conservativeResize(numRows, numCols);
    }

    template<int Dim1, int Dim2>
    inline double computeDistanceC2(std::vector<libRSF::GaussianComponent<Dim1>> pdf1, std::vector<libRSF::GaussianComponent<Dim2>> pdf2) {
        // An Analytic Distance Metric for Gaussian Mixture Models with Application in Image Retrieval, 2005
        // G. Sfikas, C. Constantinopoulos, A. Likas, and N.P. Galatsanos

        double numerator, denominator = 0.0;
        size_t K1 = pdf1.size();
        size_t K2 = pdf2.size();

        // Enforce pdf1 #components >= pdf2 #components to fit the whole computation into one for-loop
        if (pdf1.size() < pdf2.size()) {
            std::swap(pdf1, pdf2);
            std::swap(K1, K2);
        }

        for (size_t i = 0; i < K1; ++i) {
            for (size_t j = 0; j < K1; ++j) {

                const libRSF::MatrixStatic<1, 1>::Scalar weight_1i = pdf1[i].getWeight().value();
                const libRSF::MatrixStatic<Dim1, 1> mean_1i = pdf1[i].getMean();
                const libRSF::MatrixStatic<Dim1, Dim1> cov_1i = pdf1[i].getCovariance();
                const libRSF::MatrixStatic<Dim1, Dim1> info_1i = cov_1i.inverse();
                const libRSF::MatrixStatic<1, 1>::Scalar detCov_1i = cov_1i.determinant();

                const libRSF::MatrixStatic<1, 1>::Scalar weight_1j = pdf1[j].getWeight().value();
                const libRSF::MatrixStatic<Dim1, 1> mean_1j = pdf1[j].getMean();
                const libRSF::MatrixStatic<Dim1, Dim1> cov_1j = pdf1[j].getCovariance();
                const libRSF::MatrixStatic<Dim1, Dim1> info_1j = cov_1j.inverse();
                const libRSF::MatrixStatic<1, 1>::Scalar detCov_1j = cov_1j.determinant();

                // Denominator 1
                const libRSF::MatrixStatic<1, 1>::Scalar detV_11 = (info_1i + info_1j).inverse().determinant();
                const libRSF::MatrixStatic<1, 1>::Scalar K_11 = (mean_1i.transpose() * info_1i * (mean_1i - mean_1j)
                                                                 + mean_1j.transpose() * info_1j * (mean_1j - mean_1i)).value();

                denominator += weight_1i * weight_1j * std::sqrt(detV_11 / (std::exp(K_11) * detCov_1i * detCov_1j));

                // Numerator
                if (j < K2) {
                    const libRSF::MatrixStatic<1, 1>::Scalar weight_2j = pdf2[j].getWeight().value();
                    const libRSF::MatrixStatic<Dim2, 1> mean_2j = pdf2[j].getMean();
                    const libRSF::MatrixStatic<Dim2, Dim2> cov_2j = pdf2[j].getCovariance();
                    const libRSF::MatrixStatic<Dim2, Dim2> info_2j = cov_2j.inverse();
                    const libRSF::MatrixStatic<1, 1>::Scalar detCov_2j = cov_2j.determinant();

                    const libRSF::MatrixStatic<1, 1>::Scalar detV_12 = (info_1i + info_2j).inverse().determinant();
                    const libRSF::MatrixStatic<1, 1>::Scalar K_12 = (mean_1i.transpose() * info_1i * (mean_1i - mean_2j)
                                                                     + mean_2j.transpose() * info_2j * (mean_2j - mean_1i)).value();

                    numerator += weight_1i * weight_2j * std::sqrt(detV_12 / (std::exp(K_12) * detCov_1i * detCov_2j));

                    // Denominator 2
                    if (i < K2) {
                        const libRSF::MatrixStatic<1, 1>::Scalar weight_2i = pdf2[i].getWeight().value();
                        const libRSF::MatrixStatic<Dim2, 1> mean_2i = pdf2[i].getMean();
                        const libRSF::MatrixStatic<Dim2, Dim2> cov_2i = pdf2[i].getCovariance();
                        const libRSF::MatrixStatic<Dim2, Dim2> info_2i = cov_2i.inverse();
                        const libRSF::MatrixStatic<1, 1>::Scalar detCov_2i = cov_2i.determinant();

                        const libRSF::MatrixStatic<1, 1>::Scalar detV_22 = (info_2i + info_2j).inverse().determinant();
                        const libRSF::MatrixStatic<1, 1>::Scalar K_22 = (mean_2i.transpose() * info_2i * (mean_2i - mean_2j)
                                                                         + mean_2j.transpose() * info_2j * (mean_2j - mean_2i)).value();

                        denominator += weight_2i * weight_2j * std::sqrt(detV_22 / (std::exp(K_22) * detCov_2i * detCov_2j));
                    }
                }
            }
        }
        return -1.0 * std::log(2 * numerator / denominator);
    }

    template<int Dim1, int Dim2>
    inline double computeDistanceC2(fgo::inference::GaussianMixture<Dim1> pdf1, fgo::inference::GaussianMixture<Dim2> pdf2) {
        return computeDistanceC2(pdf1.getRawMixture(), pdf2.getRawMixture());
    }

    template<int Dim>
    class GaussianMixture : public libRSF::GaussianMixture<Dim> {
    public:

        /** config for mixture estimation */
        struct EstimationConfig
        {
            using EstimationAlgorithmType = fgo::inference::ErrorModelTuningType;
            /** estimation algorithm */
            EstimationAlgorithmType EstimationAlgorithm = EstimationAlgorithmType::EM;

            /** mixture properties */
            bool EstimateMean = true;

            /** termination criteria */
            int MaxIterations = 100;
            double MinLikelihoodChange = 1e-5;

            /** refuse estimation with less samples*/
            double MinimalSamples = 10;

            /** hyper-priors */
            double PriorDirichletConcentration = 1e-4;

            double PriorNormalInfoScaling = 1e-6;
            libRSF::VectorStatic<Dim> PriorNormalMean = libRSF::VectorStatic<Dim>::Zero();

            double PriorWishartDOF = Dim + 1;
            libRSF::MatrixStatic<Dim, Dim> PriorWishartScatter = libRSF::MatrixStatic<Dim, Dim>::Identity();

            /** remove components if the number of assigned samples is too low */
            bool RemoveSmallComponents = false;
            /** the minimal number of samples that have to support a component*/
            double MinSamplePerComponent = Dim + 1;

            /** check the Bhattacharyya distance between all components to merge similar ones */
            bool MergeSimilarComponents = false;
            double MergingThreshold = 0.1;
        };

        void printParameters(const rclcpp::Node::SharedPtr &nodePtr) {
            std::vector<libRSF::GaussianComponent<Dim>> Mixture;
            this->getMixture(Mixture);
            int M = Mixture.size();
            RCLCPP_INFO_STREAM(nodePtr->get_logger(), "GMM Parameter: Mean       SqrtInfo-Diagonal       Weight");
            for (int i = 0; i < M; ++i) {
                RCLCPP_INFO_STREAM(nodePtr->get_logger(), "Component " << i + 1 << ":   " <<
                                                                       Mixture.at(i).getMean().transpose() << "       "
                                                                       <<
                                                                       Mixture.at(
                                                                               i).getSqrtInformation().diagonal().transpose()
                                                                       << "  " <<
                                                                       Mixture.at(i).getWeight());
            }
        }

        bool estimateOL(const libRSF::MatrixStatic<Dim, Eigen::Dynamic> &DataMatrix,
                          const typename fgo::inference::GaussianMixture<Dim>::EstimationConfig &Config) {
            const int N = DataMatrix.cols();

            /** check size */
            if (N < Config.MinimalSamples) {
                PRINT_WARNING("Sample Size to low: ", N);
                return false;
            }

            /** set a good covariance prior, that represents the data's uncertainty*/
            auto ModifiedConfig = copyConfig(Config);
            if (Config.EstimationAlgorithm != fgo::inference::ErrorModelTuningType::EM) {
                ModifiedConfig.PriorWishartScatter =
                        libRSF::EstimateSampleCovariance(DataMatrix) * Config.PriorWishartDOF;
            }

            /** init*/
            typename libRSF::GaussianMixture<Dim>::BayesianState VBIState;
            bool ReachedMaxIteration = false;
            bool Converged = false;
            bool Merged = false;
            bool Pruned = false;
            double LikelihoodSumOld = 0;
            double LikelihoodSum = 0;
            int k = 1;

            /** iterate until convergence */
            while ((!Converged && !ReachedMaxIteration) || Merged || Pruned) {
                /** init required variables */
                std::vector<libRSF::GaussianComponent<Dim>> Mixture;
                this->getMixture(Mixture);
                int M = Mixture.size();

                switch (Config.EstimationAlgorithm) {
                    case fgo::inference::ErrorModelTuningType::EM: {
                        /** E-step */
                        libRSF::Matrix Probability(M, N);
                        LikelihoodSum = this->computeProbability(DataMatrix, Probability);

                        /** M-Step maximum likelihood */
                        this->computeMixtureParameters(DataMatrix, Probability, ModifiedConfig);
                    }
                        break;

                    case fgo::inference::ErrorModelTuningType::EM_MAP: {
                        /** E-step */
                        libRSF::Matrix Probability(M, N);
                        LikelihoodSum = this->computeProbability(DataMatrix, Probability);

                        /** M-Step maximum-a-posteriori*/
                        this->computeMixtureParametersMAP(DataMatrix, Probability, ModifiedConfig);
                    }
                        break;

                    case fgo::inference::ErrorModelTuningType::VBI: {
                        /** multivariate VBI, without Dirichlet prior*/
                        if (k == 1) {
                            /** first likelihood is not variational */
                            this->computeProbability(DataMatrix, VBIState.Responsibilities);
                        }
                        LikelihoodSum = this->doVariationalStepOL(DataMatrix, VBIState, ModifiedConfig);
                    }
                        break;

                    case fgo::inference::ErrorModelTuningType::VBI_Full: {
                        /** multivariate VBI, with Dirichlet prior*/
                        if (k == 1) {
                            /** first likelihood is not variational */
                            this->computeProbability(DataMatrix, VBIState.Responsibilities);
                        }
                        LikelihoodSum = this->doVariationalStepFullOL(DataMatrix, VBIState, ModifiedConfig);
                    }
                        break;
                    case fgo::inference::ErrorModelTuningType::VBI_ParameterLearning: {
                        /** multivariate VBI, without Dirichlet prior*/
                        if (k == 1) {
                            /** first likelihood is not variational */
                            this->computeProbability(DataMatrix, VBIState.Responsibilities);
                            LikelihoodSum = this->doVariationalStepParameterLearning(DataMatrix, VBIState, ModifiedConfig, true);
                        }
                        LikelihoodSum = this->doVariationalStepParameterLearning(DataMatrix, VBIState, ModifiedConfig);
                    }
                        break;

                    default:
                        PRINT_ERROR("Wrong mixture estimation algorithm type!");
                        break;
                }

                /** post-process mixture (only for EM, for VBI it is done internally)*/
                if (Config.EstimationAlgorithm == fgo::inference::ErrorModelTuningType::EM ||
                    Config.EstimationAlgorithm == fgo::inference::ErrorModelTuningType::EM_MAP) {
                    /** remove components with a low weight */
                    if (Config.RemoveSmallComponents) {
                        Pruned = this->prunMixture(std::min(Config.MinSamplePerComponent / N, 1.0));
                    } else {
                        Pruned = this->prunMixture(0.0);
                    }

                    /** merge components that are close to each other */
                    if (Config.MergeSimilarComponents) {
                        Merged = this->reduceMixture(Config.MergingThreshold);
                    } else {
                        Merged = false;
                    }
                }

                /** check for convergence */
                const double LikelihoodChange = std::abs(LikelihoodSum - LikelihoodSumOld) / LikelihoodSum;
                if (k > 1 && LikelihoodChange < Config.MinLikelihoodChange) {
                    Converged = true;
                } else {
                    Converged = false;
                }

                /** check for iterations */
                if (k >= Config.MaxIterations) {
                    ReachedMaxIteration = true;
                } else {
                    ReachedMaxIteration = false;
                }

                /** save for next iteration */
                LikelihoodSumOld = LikelihoodSum;

                /** increment iteration counter */
                k++;
            }

            /** take expectations from Bayesian results */
            if (Config.EstimationAlgorithm == fgo::inference::ErrorModelTuningType::VBI ||
                Config.EstimationAlgorithm == fgo::inference::ErrorModelTuningType::VBI_Full ||
                Config.EstimationAlgorithm == fgo::inference::ErrorModelTuningType::VBI_ParameterLearning) {
                this->extractParameterFromBayes(VBIState);
            }

            /** check if any GMM parameter is consistent after the estimation (only in debug mode)*/
            bool GMMConsistency = true;
            return GMMConsistency;
        }

        double doVariationalStepParameterLearning(const typename libRSF::GaussianMixture<Dim>::ErrorMatType &DataMatrix,
                                                  typename libRSF::GaussianMixture<Dim>::BayesianState &VBIState,
                                                  const typename libRSF::GaussianMixture<Dim>::EstimationConfig &Config,
                                                  bool resetPriors = false) {

            /** Implementation of:
             * Corduneanu, A. & Bishop, C.
             * Variational Bayesian Model Selection for Mixture Distributions
             * Proc. of Intl. Conf. on Artificial Intelligence and Statistics (AISTATS)
             * 2001
             */

            /** rename hyperprior variables */
            static libRSF::MatrixStatic<Dim, Dim> V0 = Config.PriorWishartScatter;
            const libRSF::MatrixStatic<Dim, Dim> Beta0 =
                    Config.PriorNormalInfoScaling * libRSF::MatrixStatic<Dim, Dim>::Identity();

            /** reset the priors to its hyperprior values in the first iteration for each inference run */
            if (resetPriors) {
                V0 = Config.PriorWishartScatter;
                // Beta0 = Config.PriorNormalInfoScaling * libRSF::MatrixStatic<Dim, Dim>::Identity();
                return -1.0;
            }

            const libRSF::VectorStatic<Dim> Mean0 = Config.PriorNormalMean;
            const double Nu0 = Config.PriorWishartDOF;

            std::vector<libRSF::GaussianComponent<Dim>> Mixture;
            this->getMixture(Mixture);

            /** adapt to current size */
            const int SampleSize = DataMatrix.cols();
            if (VBIState.Weight.size() < VBIState.Responsibilities.rows()) {
                for (int n = VBIState.Weight.size(); n < VBIState.Responsibilities.rows(); ++n) {
                    VBIState.Weight.push_back(Mixture.at(n).getWeight()(0));
                    VBIState.InfoMean.push_back(Beta0); /**< will be overwritten */
                    VBIState.MeanMean.push_back(Mean0); /**< will be overwritten */
                    VBIState.NuInfo.push_back(Nu0);
                    VBIState.WInfo.push_back(libRSF::Inverse(V0));
                }
            }
            const int GMMSize = VBIState.Weight.size();

            /** Expectation step */

            /** pre-calculate some multi-use variables */
            const libRSF::Vector SumLike = VBIState.Responsibilities.rowwise().sum();
            const libRSF::MatrixStatic<Eigen::Dynamic, Dim> SumLikeX =
                    VBIState.Responsibilities * DataMatrix.transpose();

            libRSF::MatrixVectorSTL<Dim, Dim> XX;
            for (int n = 0; n < SampleSize; n++) {
                XX.push_back(DataMatrix.col(n) * DataMatrix.col(n).transpose());
            }

            libRSF::MatrixVectorSTL<Dim, Dim> SumLikeXX;
            for (int m = 0; m < GMMSize; m++) {
                libRSF::MatrixStatic<Dim, Dim> SumLikeXXComp = libRSF::MatrixStatic<Dim, Dim>::Zero();
                for (int n = 0; n < SampleSize; n++) {
                    SumLikeXXComp += XX.at(n) * VBIState.Responsibilities(m, n);
                }
                SumLikeXX.push_back(SumLikeXXComp);
            }

            libRSF::MatrixStatic<Dim, Dim> SumMuMuEx = libRSF::MatrixStatic<Dim, Dim>::Zero();
            libRSF::MatrixStatic<Dim, Dim> SumInfoEx = libRSF::MatrixStatic<Dim, Dim>::Zero();

            /** iterate over GMM components */
            for (int m = 0; m < GMMSize; ++m) {
                /** <T_i> */
                libRSF::MatrixStatic<Dim, Dim> InfoEx = VBIState.NuInfo.at(m) * VBIState.WInfo.at(m);

                /** update mean posterior */
                VBIState.InfoMean.at(m) = Beta0 + InfoEx * SumLike(m);

                if (Config.EstimateMean == true) {
                    VBIState.MeanMean.at(m) =
                            libRSF::Inverse(VBIState.InfoMean.at(m)) * InfoEx * SumLikeX.row(m).transpose();
                }

                /** update variance posterior */
                VBIState.NuInfo.at(m) = Nu0 + SumLike(m);

                /** <Mu_i*Mu_i^T> */
                const libRSF::MatrixStatic<Dim, Dim> MuMuEx = libRSF::Inverse(VBIState.InfoMean.at(m)) +
                                                              VBIState.MeanMean.at(m) *
                                                              VBIState.MeanMean.at(m).transpose();

                SumMuMuEx += MuMuEx;

                libRSF::MatrixStatic<Dim, Dim> SumXPMu = libRSF::MatrixStatic<Dim, Dim>::Zero();
                for (int n = 0; n < SampleSize; n++) {
                    SumXPMu +=
                            DataMatrix.col(n) * VBIState.Responsibilities(m, n) * VBIState.MeanMean.at(m).transpose();
                }

                VBIState.WInfo.at(m) = (V0
                                        + SumLikeXX.at(m)
                                        - SumXPMu
                                        - VBIState.MeanMean.at(m) * SumLikeX.row(m)
                                        + MuMuEx * SumLike(m)).inverse();

                /** variational likelihood */
                /** <ln|T_i|> */
                double LogInfoEx = 0;
                for (int d = 1; d <= Dim; d++) {
                    LogInfoEx += Eigen::numext::digamma(0.5 * (VBIState.NuInfo.at(m) + 1 - d));
                }
                LogInfoEx += Dim * log(2.0) + log(VBIState.WInfo.at(m).determinant());

                /** update <T_i> */
                InfoEx = VBIState.NuInfo.at(m) * VBIState.WInfo.at(m);
                SumInfoEx += InfoEx;

                /** <Mu_i> */
                const libRSF::VectorStatic<Dim> MeanEx = VBIState.MeanMean.at(m);

                VBIState.Responsibilities.row(m).fill(0.5 * LogInfoEx + log(VBIState.Weight.at(m)));
                for (int n = 0; n < SampleSize; n++) {
                    VBIState.Responsibilities(m, n) -= 0.5 * (InfoEx * (XX.at(n)
                                                                        - MeanEx * DataMatrix.col(n).transpose()
                                                                        - DataMatrix.col(n) * MeanEx.transpose()
                                                                        + MuMuEx)).trace();
                }
                VBIState.Responsibilities.row(m) = VBIState.Responsibilities.row(m).array().exp();
            }

            /** "Maximization step" --> update probability */
            /** remove NaN */
            VBIState.Responsibilities = (VBIState.Responsibilities.array().isFinite()).select(VBIState.Responsibilities,
                                                                                              0.0);

            /** calculate relative likelihood  */
            double LikelihoodSum = VBIState.Responsibilities.sum();
            for (int n = 0; n < static_cast<int>(VBIState.Responsibilities.cols()); ++n) {
                VBIState.Responsibilities.col(n) /= VBIState.Responsibilities.col(n).sum();
            }

            /** remove NaN again (occur if sum of likelihoods is zero) */
            VBIState.Responsibilities = (VBIState.Responsibilities.array().isFinite()).select(VBIState.Responsibilities,
                                                                                              1.0 / GMMSize);

            /** calculate weights */
            for (int m = 0; m < GMMSize; ++m) {
                VBIState.Weight.at(m) = VBIState.Responsibilities.row(m).sum() / VBIState.Responsibilities.cols();
            }

            /** update beta */
            // Beta0 = 2.0 * GMMSize * Dim * libRSF::Inverse(SumMuMuEx);

            /** update scale matrix */
            const libRSF::MatrixStatic<Dim, Dim> SumInfoExSum = (SumInfoEx.transpose() + SumInfoEx) / (2.0 * GMMSize * Nu0);
            const libRSF::MatrixStatic<Dim, Dim> SumInfoExSumTransposed = SumInfoExSum.transpose();
            const libRSF::MatrixStatic<Dim, Dim> V0Unconstrained = libRSF::Inverse(SumInfoExSumTransposed);
            V0 = std::pow(1.0 / V0Unconstrained.determinant(), 1.0 / Dim) * V0Unconstrained;

            /** remove useless or degenerated components */
            for (int m = GMMSize - 1; m >= 0; --m) {
                if (((VBIState.Weight.at(m) < (Config.MinSamplePerComponent / SampleSize) &&
                      Config.RemoveSmallComponents) || std::isnan(VBIState.Weight.at(m))) &&
                    VBIState.Weight.size() > 1) {
                    /** remove component from posterior states */
                    VBIState.NuInfo.erase(VBIState.NuInfo.begin() + m);
                    VBIState.WInfo.erase(VBIState.WInfo.begin() + m);
                    VBIState.InfoMean.erase(VBIState.InfoMean.begin() + m);
                    VBIState.MeanMean.erase(VBIState.MeanMean.begin() + m);
                    VBIState.Weight.erase(VBIState.Weight.begin() + m);

                    /** remove row from probability matrix */
                    libRSF::Matrix ProbTrans = VBIState.Responsibilities.transpose();
                    fgo::inference::RemoveColumn(ProbTrans, m);
                    VBIState.Responsibilities = ProbTrans.transpose();

                    /** enforce an additional iteration after removal by resetting likelihood */
                    LikelihoodSum = 1e40;
                }
            }

            return LikelihoodSum;
        }

        double doVariationalStepOL(const typename libRSF::GaussianMixture<Dim>::ErrorMatType &DataMatrix,
                                     typename libRSF::GaussianMixture<Dim>::BayesianState &VBIState,
                                     const typename libRSF::GaussianMixture<Dim>::EstimationConfig &Config) {

            /** Implementation of:
             * Corduneanu, A. & Bishop, C.
             * Variational Bayesian Model Selection for Mixture Distributions
             * Proc. of Intl. Conf. on Artificial Intelligence and Statistics (AISTATS)
             * 2001
             */

            /** rename hyperprior variables */
            const libRSF::MatrixStatic<Dim, Dim> V0 = Config.PriorWishartScatter;
            const libRSF::MatrixStatic<Dim, Dim> Beta0 =
                    Config.PriorNormalInfoScaling * libRSF::MatrixStatic<Dim, Dim>::Identity();
            const libRSF::VectorStatic<Dim> Mean0 = Config.PriorNormalMean;
            const double Nu0 = Config.PriorWishartDOF;

            std::vector<libRSF::GaussianComponent<Dim>> Mixture;
            this->getMixture(Mixture);

            /** adapt to current size */
            const int SampleSize = DataMatrix.cols();
            if (VBIState.Weight.size() < VBIState.Responsibilities.rows()) {
                for (int n = VBIState.Weight.size(); n < VBIState.Responsibilities.rows(); ++n) {
                    VBIState.Weight.push_back(Mixture.at(n).getWeight()(0));
                    VBIState.InfoMean.push_back(Beta0); /**< will be overwritten */
                    VBIState.MeanMean.push_back(Mean0); /**< will be overwritten */
                    VBIState.NuInfo.push_back(Nu0);
                    VBIState.WInfo.push_back(libRSF::Inverse(V0));
                }
            }
            const int GMMSize = VBIState.Weight.size();

            /** Expectation step */

            /** pre-calculate some multi-use variables */
            const libRSF::Vector SumLike = VBIState.Responsibilities.rowwise().sum();
            const libRSF::MatrixStatic<Eigen::Dynamic, Dim> SumLikeX =
                    VBIState.Responsibilities * DataMatrix.transpose();

            libRSF::MatrixVectorSTL<Dim, Dim> XX;
            for (int n = 0; n < SampleSize; n++) {
                XX.push_back(DataMatrix.col(n) * DataMatrix.col(n).transpose());
            }

            libRSF::MatrixVectorSTL<Dim, Dim> SumLikeXX;
            for (int m = 0; m < GMMSize; m++) {
                libRSF::MatrixStatic<Dim, Dim> SumLikeXXComp = libRSF::MatrixStatic<Dim, Dim>::Zero();
                for (int n = 0; n < SampleSize; n++) {
                    SumLikeXXComp += XX.at(n) * VBIState.Responsibilities(m, n);
                }
                SumLikeXX.push_back(SumLikeXXComp);
            }

            /** iterate over GMM components */
            for (int m = 0; m < GMMSize; ++m) {
                /** <T_i> */
                libRSF::MatrixStatic<Dim, Dim> InfoEx = VBIState.NuInfo.at(m) * VBIState.WInfo.at(m);

                /** update mean posterior */
                VBIState.InfoMean.at(m) = Beta0 + InfoEx * SumLike(m);

                if (Config.EstimateMean == true) {
                    VBIState.MeanMean.at(m) =
                            libRSF::Inverse(VBIState.InfoMean.at(m)) * InfoEx * SumLikeX.row(m).transpose();
                }

                /** update variance posterior */
                VBIState.NuInfo.at(m) = Nu0 + SumLike(m);

                /** <Mu_i*Mu_i^T> */
                const libRSF::MatrixStatic<Dim, Dim> MuMuEx = libRSF::Inverse(VBIState.InfoMean.at(m)) +
                                                              VBIState.MeanMean.at(m) *
                                                              VBIState.MeanMean.at(m).transpose();

                libRSF::MatrixStatic<Dim, Dim> SumXPMu = libRSF::MatrixStatic<Dim, Dim>::Zero();
                for (int n = 0; n < SampleSize; n++) {
                    SumXPMu +=
                            DataMatrix.col(n) * VBIState.Responsibilities(m, n) * VBIState.MeanMean.at(m).transpose();
                }

                VBIState.WInfo.at(m) = (V0
                                        + SumLikeXX.at(m)
                                        - SumXPMu
                                        - VBIState.MeanMean.at(m) * SumLikeX.row(m)
                                        + MuMuEx * SumLike(m)).inverse();

                /** variational likelihood */
                /** <ln|T_i|> */
                double LogInfoEx = 0;
                for (int d = 1; d <= Dim; d++) {
                    LogInfoEx += Eigen::numext::digamma(0.5 * (VBIState.NuInfo.at(m) + 1 - d));
                }
                LogInfoEx += Dim * log(2.0) + log(VBIState.WInfo.at(m).determinant());

                /** update <T_i> */
                InfoEx = VBIState.NuInfo.at(m) * VBIState.WInfo.at(m);

                /** <Mu_i> */
                const libRSF::VectorStatic<Dim> MeanEx = VBIState.MeanMean.at(m);

                VBIState.Responsibilities.row(m).fill(0.5 * LogInfoEx + log(VBIState.Weight.at(m)));
                for (int n = 0; n < SampleSize; n++) {
                    VBIState.Responsibilities(m, n) -= 0.5 * (InfoEx * (XX.at(n)
                                                                        - MeanEx * DataMatrix.col(n).transpose()
                                                                        - DataMatrix.col(n) * MeanEx.transpose()
                                                                        + MuMuEx)).trace();
                }
                VBIState.Responsibilities.row(m) = VBIState.Responsibilities.row(m).array().exp();
            }

            /** "Maximization step" --> update probability */
            /** remove NaN */
            VBIState.Responsibilities = (VBIState.Responsibilities.array().isFinite()).select(VBIState.Responsibilities,
                                                                                              0.0);

            /** calculate relative likelihood  */
            double LikelihoodSum = VBIState.Responsibilities.sum();
            for (int n = 0; n < static_cast<int>(VBIState.Responsibilities.cols()); ++n) {
                VBIState.Responsibilities.col(n) /= VBIState.Responsibilities.col(n).sum();
            }

            /** remove NaN again (occur if sum of likelihoods is zero) */
            VBIState.Responsibilities = (VBIState.Responsibilities.array().isFinite()).select(VBIState.Responsibilities,
                                                                                              1.0 / GMMSize);

            /** calculate weights */
            for (int m = 0; m < GMMSize; ++m) {
                VBIState.Weight.at(m) = VBIState.Responsibilities.row(m).sum() / VBIState.Responsibilities.cols();
            }

            /** remove useless or degenerated components */
            for (int m = GMMSize - 1; m >= 0; --m) {
                if (((VBIState.Weight.at(m) < (Config.MinSamplePerComponent / SampleSize) &&
                      Config.RemoveSmallComponents) || std::isnan(VBIState.Weight.at(m))) &&
                    VBIState.Weight.size() > 1) {
                    /** remove component from posterior states */
                    VBIState.NuInfo.erase(VBIState.NuInfo.begin() + m);
                    VBIState.WInfo.erase(VBIState.WInfo.begin() + m);
                    VBIState.InfoMean.erase(VBIState.InfoMean.begin() + m);
                    VBIState.MeanMean.erase(VBIState.MeanMean.begin() + m);
                    VBIState.Weight.erase(VBIState.Weight.begin() + m);

                    /** remove row from probability matrix */
                    libRSF::Matrix ProbTrans = VBIState.Responsibilities.transpose();
                    fgo::inference::RemoveColumn(ProbTrans, m);
                    VBIState.Responsibilities = ProbTrans.transpose();

                    /** enforce an additional iteration after removal by resetting likelihood */
                    LikelihoodSum = 1e40;
                }
            }

            return LikelihoodSum;
        }

        double doVariationalStepFullOL(const typename libRSF::GaussianMixture<Dim>::ErrorMatType &DataMatrix,
                                         typename libRSF::GaussianMixture<Dim>::BayesianState &VBIState,
                                         const typename libRSF::GaussianMixture<Dim>::EstimationConfig &Config) {
            /** Implementation based on:
            * Christopher M. Bishop
            * Pattern Recognition and Machine Learning, Section 10.2
            * 2006
            */

            /** copy hyper-priors */
            const libRSF::MatrixStatic<Dim, Dim> W0Inv = Config.PriorWishartScatter;
            const double Beta0 = Config.PriorNormalInfoScaling;
            const double Nu0 = Config.PriorWishartDOF;
            const double Alpha0 = Config.PriorDirichletConcentration;
            const libRSF::VectorStatic<Dim> Mean0 = Config.PriorNormalMean;

            std::vector<libRSF::GaussianComponent<Dim>> Mixture;
            this->getMixture(Mixture);

            /** adapt state to current size */
            const int SampleSize = DataMatrix.cols();
            if (VBIState.AlphaWeight.size() < VBIState.Responsibilities.rows()) {
                for (int n = VBIState.AlphaWeight.size(); n < VBIState.Responsibilities.rows(); ++n) {
                    VBIState.AlphaWeight.push_back(Alpha0);
                    VBIState.MeanMean.push_back(Mean0);
                    VBIState.BetaMean.push_back(Beta0);
                    VBIState.NuInfo.push_back(Nu0);
                    VBIState.WInfo.push_back(libRSF::Inverse(Config.PriorWishartScatter));
                    VBIState.Weight.push_back(Mixture.at(n).getWeight()(0));
                }
            }
            const int GMMSize = VBIState.AlphaWeight.size();

            /** pre-calculate useful variables */
            const libRSF::Vector N = VBIState.Responsibilities.rowwise().sum();
            const double NSum = N.sum();
            const libRSF::Vector NInv = (N.array().inverse().isFinite()).select(N.array().inverse(),
                                                                                0.0); /**< catch NaN */
            libRSF::MatrixStatic<Eigen::Dynamic, Dim> MeanX(GMMSize, Dim);
            for (int k = 0; k < GMMSize; k++) {
                MeanX.row(k) = NInv(k) * (DataMatrix * VBIState.Responsibilities.row(k).asDiagonal()).rowwise().sum();
            }

            libRSF::MatrixVectorSTL<Dim, Dim> S;
            for (int k = 0; k < GMMSize; k++) {
                libRSF::MatrixStatic<Dim, Dim> Sum = libRSF::MatrixStatic<Dim, Dim>::Zero();
                for (int n = 0; n < SampleSize; n++) {
                    Sum += VBIState.Responsibilities(k, n)
                           * ((DataMatrix.col(n) - MeanX.row(k).transpose())
                              * (DataMatrix.col(n) - MeanX.row(k).transpose()).transpose());
                }
                S.push_back(NInv(k) * Sum);
            }

            /** update posteriors */
            double SumAlpha = 0;
            for (int k = 0; k < GMMSize; k++) {
                VBIState.AlphaWeight.at(k) = Alpha0 + N(k);
                SumAlpha += VBIState.AlphaWeight.at(k);

                VBIState.BetaMean.at(k) = Beta0 + N(k);
                VBIState.MeanMean.at(k) =
                        1.0 / VBIState.BetaMean.at(k) * (Beta0 * Mean0 + N(k) * MeanX.row(k).transpose());

                const libRSF::MatrixStatic<Dim, Dim> WInfoInv = W0Inv
                                                                + N(k) * S.at(k)
                                                                + Beta0 * N(k) / (Beta0 + N(k)) *
                                                                  (MeanX.row(k).transpose() - Mean0) *
                                                                  (MeanX.row(k).transpose() - Mean0).transpose();

                VBIState.WInfo.at(k) = libRSF::Inverse(WInfoInv);
                VBIState.NuInfo.at(k) = Nu0 + N(k);

                /** update expected weight */
                VBIState.Weight.at(k) = (Alpha0 + N(k)) / (GMMSize * Alpha0 + NSum);
            }

            /** evaluate expectations */
            libRSF::Vector ExpLnInfo(GMMSize);
            libRSF::Vector ExpLnWeight(GMMSize);
            for (int k = 0; k < GMMSize; k++) {
                double Sum = 0;
                for (int d = 1; d <= Dim; d++) {
                    Sum += Eigen::numext::digamma(0.5 * (VBIState.NuInfo.at(k) + 1 - d));
                }
                ExpLnInfo(k) = Sum + Dim * log(2.0) + log(VBIState.WInfo.at(k).determinant());
                ExpLnWeight(k) = Eigen::numext::digamma(VBIState.AlphaWeight.at(k))
                                 - Eigen::numext::digamma(SumAlpha);
            }

            /** evaluate responsibilities */
            for (int k = 0; k < GMMSize; k++) {
                for (int n = 0; n < SampleSize; n++) {
                    VBIState.Responsibilities(k, n) = exp(ExpLnWeight(k)
                                                          + ExpLnInfo(k) / 2.0
                                                          - Dim / (2.0 * VBIState.BetaMean.at(k))
                                                          - VBIState.NuInfo.at(k) / 2.0
                                                            *
                                                            ((DataMatrix.col(n) - VBIState.MeanMean.at(k)).transpose() *
                                                             VBIState.WInfo.at(k)).dot(
                                                                    (DataMatrix.col(n) - VBIState.MeanMean.at(k))));
                }
            }

            /** catch numerical issues with the exp() */
            VBIState.Responsibilities = (VBIState.Responsibilities.array().isFinite()).select(VBIState.Responsibilities,
                                                                                              0.0);

            /** save likelihood before(!) it is normalized as probability */
            double LikelihoodSum = VBIState.Responsibilities.sum();

            /** normalize */
            for (int n = 0; n < static_cast<int>(VBIState.Responsibilities.cols()); ++n) {
                VBIState.Responsibilities.col(n) /= VBIState.Responsibilities.col(n).sum();
            }

            /** remove NaN (occur if sum of likelihoods is zero) */
            VBIState.Responsibilities = (VBIState.Responsibilities.array().isFinite()).select(VBIState.Responsibilities,
                                                                                              1.0 / GMMSize);

            /** remove useless or degenerated components */
            for (int k = GMMSize - 1; k >= 0; --k) {

                if (((VBIState.Weight.at(k) < (Config.MinSamplePerComponent / SampleSize) &&
                      Config.RemoveSmallComponents) || std::isnan(VBIState.Weight.at(k))) &&
                    VBIState.Weight.size() > 1) {
                    /** remove component from posterior states */
                    VBIState.AlphaWeight.erase(VBIState.AlphaWeight.begin() + k);
                    VBIState.MeanMean.erase(VBIState.MeanMean.begin() + k);
                    VBIState.BetaMean.erase(VBIState.BetaMean.begin() + k);
                    VBIState.NuInfo.erase(VBIState.NuInfo.begin() + k);
                    VBIState.WInfo.erase(VBIState.WInfo.begin() + k);
                    VBIState.Weight.erase(VBIState.Weight.begin() + k);

                    /** remove row from probability matrix */
                    libRSF::Matrix RespTrans = VBIState.Responsibilities.transpose();
                    fgo::inference::RemoveColumn(RespTrans, k);
                    VBIState.Responsibilities = RespTrans.transpose();

                    /** enforce an additional iteration after removal by resetting likelihood */
                    LikelihoodSum = 1e40;
                }
            }

            return LikelihoodSum;
        }

        /** improved version */
        libRSF::VectorStatic<Dim> removeOffsetOL()
        {
            if (Dim > 1)
                return {};
            const int NumberOfComponents = this->getNumberOfComponents();
            const double MinimumWeight = 1.0/ this->Mixture_.size()*0.8;

            /** remove offset of the first "LOS" component */
            this->sortComponentsByMean();
            libRSF::VectorStatic<Dim> MeanLOS;
            for(int i = 0; i < NumberOfComponents; ++i)
            {
                if(this->Mixture_.at(i).getWeight()(0) >= MinimumWeight)
                {
                    MeanLOS = this->Mixture_.at(i).getMean();
                    break;
                }
            }

            this->removeGivenOffset(MeanLOS);
            return MeanLOS;
        }

        libRSF::VectorStatic<Dim> getDominantModeMean() {

            this->sortComponentsByWeight();
            libRSF::VectorStatic<Dim> MeanLOS = this->Mixture_.front().getMean();
            return MeanLOS;
        }

        /** improved version */
        libRSF::VectorStatic<Dim> removeOffsetExperimental(fgo::inference::GaussianMixture<Dim> &other)
        {
            const auto C2 = computeDistanceC2(*this, other);
            RCLCPP_INFO_STREAM(rclcpp::get_logger("GAUSSIAN MIXTURE DEBUG"), "C2 = " << C2);

            if (C2 > 0.1)
                return {};

            if (Dim > 1)
                return {};


            /** remove offset of the first "LOS" component */
            this->sortComponentsByWeight();
            libRSF::VectorStatic<Dim> MeanLOS = this->Mixture_.front().getMean();

            const auto diffMeanLOS = MeanLOS - other.getDominantModeMean();

            this->removeGivenOffset(diffMeanLOS);

            RCLCPP_INFO_STREAM(rclcpp::get_logger("GAUSSIAN MIXTURE DEBUG"), "diffMeanLOS = " << diffMeanLOS);

            return diffMeanLOS;
        }

        std::vector<libRSF::GaussianComponent<Dim>> getRawMixture() const {
            return this->Mixture_;
        }

        /** sampling */
        libRSF::VectorVectorSTL<Dim> DrawSamples(const int Number, int MinSamples = 0)
        {
            /** crate storage */
            libRSF::VectorVectorSTL<Dim> SampleVector;
            int AggregatedOffset = 0;

            /** loop over components and draw samples */
            const int NumComp = this->getNumberOfComponents();

            if (MinSamples >= Number/NumComp)
                MinSamples = Number / NumComp;

            std::sort(this->Mixture_.begin(), this->Mixture_.end(), [](const libRSF::GaussianComponent<Dim> &lhs, const libRSF::GaussianComponent<Dim> &rhs) -> bool {
                return lhs.getWeight()(0) < rhs.getWeight()(0);
            });

            for (int n = 0; n < NumComp; n++)
            {
                /** use weight of components for number of samples*/
                int CurrentNumber;
                if (n < NumComp-1)
                {
                    CurrentNumber = round(Number * this->Mixture_.at(n).getWeight()(0)) - round((double) AggregatedOffset / (NumComp - n));
                    if (CurrentNumber < MinSamples) {
                        AggregatedOffset += CurrentNumber - MinSamples;
                        CurrentNumber = MinSamples;
                    }
                }
                else
                {
                    /** fill with last component to avoid round-off */
                    CurrentNumber = Number - SampleVector.size();
                }

                /** draw */
                const libRSF::VectorVectorSTL<Dim> CurrentSamples = this->Mixture_.at(n).DrawSamples(CurrentNumber);

                /** append */
                SampleVector.insert(std::end(SampleVector), std::begin(CurrentSamples), std::end(CurrentSamples));
            }

            return SampleVector;
        }

        typename std::vector<libRSF::GaussianComponent<Dim>>::iterator begin() {
            return this->Mixture_.begin();
        }

        typename std::vector<libRSF::GaussianComponent<Dim>>::iterator end() {
            return this->Mixture_.end();
        }

    private:
        static typename libRSF::GaussianMixture<Dim>::EstimationConfig copyConfig(typename fgo::inference::GaussianMixture<Dim>::EstimationConfig fgoConfig) {

            typename libRSF::GaussianMixture<Dim>::EstimationConfig librsfConfig;

            librsfConfig.EstimationAlgorithm = libRSF::ErrorModelTuningType::None;
            librsfConfig.EstimateMean = fgoConfig.EstimateMean;
            librsfConfig.MaxIterations = fgoConfig.MaxIterations;
            librsfConfig.MinLikelihoodChange = fgoConfig.MinLikelihoodChange;
            librsfConfig.PriorDirichletConcentration = fgoConfig.PriorDirichletConcentration;
            librsfConfig.PriorNormalInfoScaling = fgoConfig.PriorNormalInfoScaling;
            librsfConfig.PriorNormalMean = fgoConfig.PriorNormalMean;
            librsfConfig.PriorWishartDOF = fgoConfig.PriorWishartDOF;
            librsfConfig.PriorWishartScatter = fgoConfig.PriorWishartScatter;
            librsfConfig.RemoveSmallComponents = fgoConfig.RemoveSmallComponents;
            librsfConfig.MinSamplePerComponent = fgoConfig.MinSamplePerComponent;
            librsfConfig.MergeSimilarComponents = fgoConfig.MergeSimilarComponents;
            librsfConfig.MergingThreshold = fgoConfig.MergingThreshold;
            return librsfConfig;
        }
    };

}

#endif //ONLINE_FGO_GAUSSIANMIXTURE_H
