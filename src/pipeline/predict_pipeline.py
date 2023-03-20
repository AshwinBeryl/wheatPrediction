import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join('artifacts',"model.pkl")#'artifacts\model.pkl'
            preprocessor_path=os.path.join('artifacts',"preprocessor.pkl")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(self,
                area,
                perimeter,
                compactness,
                lengthOfKernel,
                widthOfKernel,
                asymmetryCoefficent,
                lengthOfKernelGroove
                ):

        self.area=area
        self.perimeter=perimeter
        self.compactness=compactness
        self.lengthOfKernel=lengthOfKernel
        self.widthOfKernel=widthOfKernel
        self.asymmetryCoefficent=asymmetryCoefficent
        self.lengthOfKernelGroove=lengthOfKernelGroove

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "area":[self.area],
                "perimeter":[self.perimeter],
                "compactness":[self.compactness],
                "lengthOfKernel":[self.lengthOfKernel],
                "widthOfKernel":[self.widthOfKernel],
                "asymmetryCoefficent":[self.asymmetryCoefficent],
                "lengthOfKernelGroove":[self.lengthOfKernelGroove]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

