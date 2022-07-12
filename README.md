# Development (TODO: Change Repo Name)

## **Description**
This repository contains the code to generate the questionnaire that was conducted for the sake of our paper *Study on the Helpfulness of Explainable Artificial Intelligence (2022)* as well as the scripts for the analysis of the gathered survey results. New randomly generated questionnaires on the basis of the chosen XAI-methods and dataset (described below) can be easily generated as well. 

In our work we specifically examined the question how far the chosen XAI-methods *Confidence Scores,* *LRP*, *gradCAM*, *Integrated Gradients*, *LIME* and *SHAP* enable a user to correctly identify whether a model (*AlexNet* or *VGG16*) classified randomly chosen images from the *imagenetv2-matched-frequency* dataset correctly. For the decision whether to trust or distrust the model, the participants were only given the generated explanation by one of the XAI-methods but not the actual predicted class of the model.\
An example question consisting of the original image alongside the explanation image generated by *gradCAM* can be seen below:

![](data/readme/question_example.png)






A complete *questionnaire* that is generated by this repository is represented by a folder containing twelve subfolders representing  different questionnaire forms. Each of those subfolders contains the actual questions (original and corresponding explanation images). For further information on the survey design and questionnaire generation procedure we refer to our paper *Study on the Helpfulness of Explainable Artificial Intelligence (2022)* section *III) Methodology D) Survey Design*.


## **Setup**
1. Clone the repository.
2. Create a virtual environment, activate it  and install the packages defined in *requirments.txt* via ```pip install -r requirements.txt```.
3. Download the *imagenetv2-matched-frequency* dataset from http://imagenetv2public.s3-website-us-west-2.amazonaws.com/ and paste the unpacked folder (leave folder name unchanged) in the *data* folder of the project structure. For more information on this dataset we refer to https://github.com/modestyachts/ImageNetV2.

## **Usage**

### **Reproduction of questionnaire (forms) used in conducted survey**
***Remark***: *All images that are generated by the following steps can be found in the directory "questionnaire_forms_conducted_survey". Thus there is no explicit need for executing the following steps if time and computational resources need to be saved.*
1. Run the main.py file to generate the questions for all methods. If questions for certain XAI-method(s) should not be generated, run the main.py file and set the corresponding parameter(s) (--LRP, --gradCam, --LIME, --IntGrad, --CS, --SAHP) to *False*. For example ```python main.py --LIME False --CS False``` will generate all questions except from *LIME* and *Confidence Scores*.
2. Find the results in a newly generated folder *questionnaire_forms_<currentDate<d>>_<currentTime<d>>* in the root directory.
3. TODO: sollen wir noch parameter einfügen der entscheidet, ob bestehende Survey oder neue Survey berechent?

### **Generation of new randomly generated questionnaire (forms)**

1. Run the *experiment_creator.py* file.\
Explanation: This will generate a new randomly drawn questionnaire plan meeting the conditions defined in our paper. The plan is a 2D-list, where each inner list represents a individual questionnaire form and contains question-tuples of the format *(img_idx, model_used, xai_used, is_pred_correct)*. The 2D-list is saved as a *.pickle* in *data/question_generation*.
2. Run the *main.py* file.\
Explanation: The questionnaire plan generated in step 1 is read and the defined questions are generated iteratively.
3. Find the results in a newly generated folder *questionnaire_forms_<currentDate<d>>_<currentTime<d>>* in the root directory.

### **Reproduction of survey analysis results**
1. Run the *questionnaire_data_preparation.ipynb*.\
Explanation: The notebook reads the raw analysis data that was downloaded from the used survey platform *SoSci*, transforms it into a suitable data format and saves the file as *.xlsx* in the *data/survey_results* folder for further analysis.
2. Run the *questionnaire_participants_statistics_charts.ipynb* notebook to see some visual statistics regarding the participants.
3. Run the *questionnaire_metrics_calculator.ipynb* notebook to calculate the metrics that are bound to the research questions defined in our paper. Hypotheses testing is done here as well.
4. TODO: Confusion Matrix??

## **Authors and acknowledgment**
Show your appreciation to those who have contributed to the project.

## **License**
For open source projects, say how it is licensed.

