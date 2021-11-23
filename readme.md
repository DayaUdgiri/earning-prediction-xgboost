This is to build XGBoost Classifier for predicting whether a person makes over
50K per year or not from classic adult dataset. 

Data Set Information:

Extraction was done by Barry Becker from the 1994 Census
database. A set of reasonably clean records was extracted using the
following conditions: ((AAGE>16) && (AGI>100) &&
(AFNLWGT>1)&& (HRSWK>0))

Attribute Information: <br>

Listing of attributes: >50K, <=50K.<br>

age: continuous.<br>

workclass: Private, Self-emp-not-inc, Self-emp-inc, Federal-gov,Local-gov, State-gov, Without-pay, Never-worked.<br>
fnlwgt: continuous.<br>

education: Bachelors, Some-college, 11th, HS-grad, Prof-school,Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th,Doctorate, 5th-6th, Preschool.<br>

education-num: continuous.<br>

marital-status: Married-civ-spouse, Divorced, Never-married,Separated, Widowed, Married-spouse-absent, Married-AF-spouse.<br>

occupation: Tech-support, Craft-repair, Other-service, Sales, Exec-managerial, Prof-specialty, Handlers-cleaners, Machine-op-inspct,Adm-clerical, Farming-fishing, Transport-moving, Priv-house-serv,Protective-serv, Armed-Forces.<br>
relationship: Wife, Own-child, Husband, Not-in-family, Other-relative,Unmarried. <br>

race: White, Asian-Pac-Islander, Amer-Indian-Eskimo, Other, Black.<br>

sex: Female, Male.<br>

capital-gain: continuous.<br>

capital-loss: continuous.<br>

hours-per-week: continuous.<br>

native-country: United-States, Cambodia, England, Puerto-Rico,Canada, Germany, Outlying-US(Guam-USVI-etc), India, Japan,
Greece, South, China, Cuba, Iran, Honduras, Philippines, Italy,Poland, Jamaica, Vietnam, Mexico, Portugal, Ireland, France,
Dominican-Republic, Laos, Ecuador, Taiwan, Haiti, Columbia,Hungary, Guatemala, Nicaragua, Scotland, Thailand, Yugoslavia, El-Salvador, Trinadad&Tobago, Peru, Hong, Holand-Netherlands.<br>


The XGboost Classifier model along with basic web frontend is deployed in Heroku to take input and provide prediction.Then basic website with flask api and deploy it along with model on Heroku.

https://XGBoost_earning_prediction.herokuapp.com/