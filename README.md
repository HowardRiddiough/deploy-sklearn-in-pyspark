# Python ML Deployment in PySpark Using Pandas UDFs
This repo includes a [notebook](deploying-python-ml-in-pyspark.ipynb) that defines a versatile python function that can be used to deploy python ml in PySpark, several examples are used to demonstrate how python ml can be deployed in PySpark:

- Deploying a RandomForestRegressor in PySpark
- Deployment of ML Pipeline that scales numerical features
- Deployment of ML Pipeline that is capable of preprocessing mixed feature types

# Introducing the *spark_predict* function: a vessle for python ml deployment in PySpark

Making predictions in PySpark using sophistaicated python ml is unlocked using our `spark_predict` function defined below.

`spark_predict` is a wrapper around a `pandas_udf`, a wrapper is used to enable a python ml model to be passed to the `pandas_udf`.

    def spark_predict(model, cols) -> pyspark.sql.column:
        """This function deploys python ml in PySpark using the `predict` method of `model.
    
        Args:
            model: python ml model with sklearn API
            cols (list-like): Features used for predictions, required to be present as columns in the spark DataFrame used to make predictions.
        """
        @sf.pandas_udf(returnType=DoubleType())
        def predict_pandas_udf(*cols):
            # cols will be a tuple of pandas.Series here.
            x = pd.concat(cols, axis=1)
            return pd.Series(model.predict(x))
    
        return predict_pandas_udf(*cols)


# Python ML Deployment in practice
The [deploying-python-ml-in-pyspark](deploying-python-ml-in-pyspark.ipynb) notebook demonstrates how `spark_predict` can be used to deploy python ML in PySpark. It is shown that `spark_predict` is capable of deploying simple ml models in addition to more sophisticated pipelines in PySpark.

I often use both categorical and numerical features in predictive model, so I have included an example that includes an sklearn `Pipeline` designed to scale numerical and encode categorical data. This particular pipeline appends two preprocessing pipelines to a random forest to create a full prediction pipeline that will transform categorical and numerical data and fit a model. And of course this pipeline is deployed in PySpark using the `spark_predict` function.

# Requirements
See [requirements.txt](requirements.txt).

# PySpark Installation
The code used in the [deploying-python-ml-in-pyspark](deploying-python-ml-in-pyspark.ipynb) notebook requires installation of PySpark. We leave the installation of PySpark for the user.

# Further Reading
 - The code used in is based on the excellent excellent blog post ["Prediction at Scale with scikit-learn and PySpark Pandas UDFs"](https://medium.com/civis-analytics/prediction-at-scale-with-scikit-learn-and-pyspark-pandas-udfs-51d5ebfb2cd8) written by **Michael Heilman**.
- [sklearn](https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer_mixed_types.html) has more information on column transformers with mixed types.
