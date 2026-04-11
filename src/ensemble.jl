using MLJ
using MLJLinearModels
using DecisionTree
using EvoTrees

# Ensure the interface is loaded for DecisionTree models
import MLJDecisionTreeInterface

"""
    superlearner(; metalearner=nothing, binary=false)

Create a SuperLearner (Stacked Ensemble) library for NPCausal.
Returns an MLJ.Stack with a diverse set of models.
"""
function superlearner(; metalearner=nothing, binary=false)
    if binary
        # Propensity Score Models (Classification)
        # We use explicit constructors where possible, but DecisionTree needs the MLJ wrapper
        RF = MLJDecisionTreeInterface.RandomForestClassifier
        
        default_metalearner = (metalearner === nothing) ? LogisticClassifier() : metalearner
        
        return Stack(
            metalearner = default_metalearner,
            resampling = CV(nfolds=5),
            glm = LogisticClassifier(),
            rf = RF(n_trees=500),
            evo = EvoTreeClassifier(nrounds=100, max_depth=5),
            mean = ConstantClassifier()
        )
    else
        # Outcome Regression Models (Regression)
        RF = MLJDecisionTreeInterface.RandomForestRegressor
        
        default_metalearner = (metalearner === nothing) ? LinearRegressor() : metalearner
        
        return Stack(
            metalearner = default_metalearner,
            resampling = CV(nfolds=5),
            glm = LinearRegressor(),
            rf = RF(n_trees=500),
            evo = EvoTreeRegressor(nrounds=100, max_depth=5),
            mean = ConstantRegressor()
        )
    end
end
