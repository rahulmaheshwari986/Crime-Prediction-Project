/*Customize the path according to where you save the csv files*/
libname project "D:\Meiyi\kaggle";

/*Defining macro path variables. Change the path according to wher you save the csv files*/
%let trainfile = "D:\Meiyi\Kaggle\train.csv";
%let testfile = "D:\Meiyi\Kaggle\test.csv";

/*Add macro variables for police station coordinates of a new district, if desired.*/
%let MISSION_PD_X = 37.763112;
%let MISSION_PD_Y = -122.422016;
%let SOUTHERN_PD_X = 37.773312;
%let SOUTHERN_PD_Y = -122.389516;

/*Macro definitions for reusable code across any district*/
/*******************************************************************************************************************/
/*This macros creates some predictor variables for the desired district*/
%macro CreatePredictorVars (indata, outdata, districtName, DISTRICT_PD_X, DISTRICT_PD_Y);
data &outdata;
set &indata;
where PdDistrict = &districtName;
Dist = geodist(X, Y, &DISTRICT_PD_X, &DISTRICT_PD_Y);  /*Distance of the place from district police station*/
run;

proc sql;
alter table &outdata
add DistMin num, DistMax num, DistStep num;

create table temp as (Select Dist from &outdata);

update &outdata
set DistMin = (select min(Dist) from temp where Dist <= 5),
	DistMax = (select max(Dist) from temp where Dist <= 5),
	DistStep = (select (max(Dist)-min(Dist))/9 from temp where Dist <= 5);
quit;

/*Most of the crime incidents were found to happen within 5 km of the district police station, 
so we filter out the few outliers and create a categorical predictor variable with 9 levels*/
data &outdata;
set &outdata;
where Dist <= 5;
if Dist < DistMin + DistStep then DisType = 0 ;
else if Dist >= DistMin + DistStep and Dist < DistMin + 2*DistStep then DisType = 1;
else if Dist >= DistMin + 2*DistStep and Dist < DistMin + 3*DistStep then DisType = 2;
else if Dist >= DistMin + 3*DistStep and Dist < DistMin + 4*DistStep then DisType = 3 ;
else if Dist >= DistMin + 4*DistStep and Dist < DistMin + 5*DistStep then DisType = 4 ;
else if Dist >= DistMin + 5*DistStep and Dist < DistMin + 6*DistStep then DisType = 5 ;
else if Dist >= DistMin + 6*DistStep and Dist < DistMin + 7*DistStep then DisType = 6 ;
else if Dist >= DistMin + 7*DistStep and Dist < DistMin + 8*DistStep then DisType = 7 ;
else if Dist >= DistMin + 8*DistStep then DisType = 8 ;

/*Categorical predictor variable denoting which time of the day the crime happened
1= morning, 2 = afternoon, 3 = evening, 4 = night*/
if HOUR(Dates) in (6:12) then TimeSeg = 1 ;
else if HOUR(Dates) in (13:18) then TimeSeg = 2 ;
else if HOUR(Dates) in (19:22) then TimeSeg = 3 ;
else if HOUR(Dates) in (23, 0:5) then TimeSeg = 4;

/*Categorical predictor variable denoting whether the crime happened on a weekday or a weekend
1 = weekday, 2 = weekend*/
if DayOfWeek in ('Monday','Tuesday','Wednesday','Thursday') then DayOfWeek = 1;
else if DayOfWeek in ('Friday','Saturday','Sunday') then DayOfWeek = 2;

/*Categorical predictor variable denoting whether the crime happened in a block or a street corner
0 - Street corner, 1 - Block*/
if PRXMATCH('/Block of/', Address)= 0 then BLorST = 1;  
else BLorST = 0;
run;

/*Cleanup the temporary datasets*/
proc datasets library=work nolist;
delete temp;  
run;
%mend CreatePredictorVars;

/*This macros creates clusters based on the XY coordinates of the crime incidents 
and assigns cluster to each observation as a categorical predictor variable*/
%macro CreateClusterVar (data);
proc sql;
alter table &data
drop DistMin, DistMax, DistStep;

create table cluster as (SELECT * From &data);
quit;

proc fastclus data=cluster maxc=8 maxiter=10 out=&data mean=clusterMeans;
var Y X;
run;

proc freq data=&data;
tables Category * Cluster / norow nocol nofreq;
run;

proc datasets library=work nolist;
delete cluster;  
run;
%mend CreateClusterVar;

/*This macro takes samples for each category of crime specified by the sample size 
and creates a sampled train dataset for use*/
%macro CreateSampleTrain (indata, outdata, sampleSize);
data personalCrime;
set &indata;
where Category = 'PERSONAL';
run;

proc surveyselect data=personalCrime 
method=srs n=&sampleSize out=samplePersonalCrime;
run;

data propertyCrime;
set &indata;
where Category = 'PROPERTY';
run;

proc surveyselect data=propertyCrime 
method=srs n=&sampleSize out=samplePropertyCrime;
run;

data statutoryCrime;
set &indata;
where Category = 'STATUTORY';
run;

proc surveyselect data=statutoryCrime
method=srs n=&sampleSize out=sampleStatutoryCrime;
run;

proc sql;
create table &outdata as  
(select * from samplePersonalCrime
 union
 select * from samplePropertyCrime
 union
 select * from sampleStatutoryCrime);
quit;

proc datasets library=work nolist;
delete personalCrime samplePersonalCrime propertyCrime samplePropertyCrime statutoryCrime sampleStatutoryCrime;  
run;
%mend CreateSampleTrain;

/*This macro runs generalized multinomial logistics regression based on a set of predictor variables 
to predict the nominal response variable of crime cSategory, and saves the model 
and also analyzies the prediction results on the sampled train dataset*/
%macro TrainModel (indata, outmodel, outprediction);
proc logistic data=&indata noprint outmodel=&outmodel;
class BLorST DisType DayOfWeek TimeSeg CLUSTER;
model Category = BLorST DisType DayOfWeek TimeSeg CLUSTER / link=glogit;
OUTPUT OUT=&outprediction PREDPROBS=INDIVIDUAL;
RUN;

proc freq data=&outprediction;
table _FROM_ * _into_  / nofreq nocol norow; 
run;
%mend;

/*This macro assigns clusters to the observations in the test dataset based on the minimum euclidean distance 
to one of the cluster means calculated for the train dataset*/
%macro AssignClusterTest (data);
proc iml;
use &data;
read all var{"Y" "X"} into A;
close data;

use clusterMeans;
read all var{"Y" "X"} into B;
close clusterMeans;

n = nrow(A); m = nrow(B);
idx = T(repeat(1:n, m));
jdx = shape(repeat(1:m, n), n);
diff = A[idx, ] - B[jdx, ];

euclideanDistMatrix = shape(sqrt(diff[, ##]), n); 
idxMin = euclideanDistMatrix[, >:<];         /* find columns for min of each row */
create temp from idxmin[colname={Cluster}];
append from idxmin;
close temp;
quit;

data &data;
merge &data temp;
run;

proc datasets library=work nolist;
delete temp clusterMeans;
run;
%mend;

/*This macro runs the trained model on the test dataset and analyses the scored results*/
%macro ScoreTest (inmodel, inTestData, outScoredTest);
proc logistic inmodel=&inmodel;
score data=&inTestData out=&outScoredTest;
run;

proc freq data=&outScoredTest;
table I_category / norow nocol;
run;
%mend;

/*******************************************************************************************************************/

/*Read the train and test datsets from the csv files*/
data project.train;
infile &trainfile dsd dlm=',' firstobs=2;
input Dates :anydtdtm21. Category :$30. Descript :$80. DayOfWeek :$10. 
PdDistrict :$15. Resolution :$40. Address :$60. Y X;
format Dates datetime.;
run;

data project.test;
infile &testfile dsd firstobs=2;
input Id Dates :anydtdtm21. DayOfWeek :$10. PdDistrict :$15. Address :$60. Y X;
format Dates datetime.;
run;

/*Here we assign the 39 crime categories into 1 of the 3 primary crime type categories, 
and also filter out several trivial categories which are either not crimes in the real sense or are too ubiquitous
to be of any utility in logistics regression modeling*/
data project.train;
SET project.train;
WHERE Category NOTIN ('MISSING PERSON', 'SECONDARY CODES', 'RECOVERED VEHICLE','RUNAWAY', 'SUICIDE', 
'TREA', 'SUSPICIOUS OCC', 'WARRANTS', 'LARCENY/THEFT', 'NON-CRIMINAL' ,'OTHER OFFENSES');

if Category IN ('ASSAULT', 'SEX OFFENSES FORCIBLE', 'KIDNAPPING', 'SEX OFFENSES NON FORCIBLE') 
then Category = 'PERSONAL';

else if Category IN('VEHICLE THEFT', 'VANDALISM', 'BURGLARY', 'ROBBERY', 'FRAUD', 'FORGERY/COUNTERFEITING',
'TRESPASS', 'STOLEN PROPERTY', 'ARSON', 'EMBEZZLEMENT', 'BAD CHECKS', 'BRIBERY', 'EXTORTION')
then Category = 'PROPERTY';

else if Category IN ('DRUG/NARCOTIC', 'WEAPON LAWS', 'PROSTITUTION', 'DISORDERLY CONDUCT', 'DRUNKENNESS',
'DRIVING UNDER THE INFLUENCE', 'LIQUOR LAWS', 'LOITERING', 'FAMILY OFFENSES', 'GAMBLING', 'PORNOGRAPHY/OBSCENE MAT')
then Category ='STATUTORY';

run;

/*Descrptive analysis of category vs. PdDistrict. We can see from the results that there is no strong corelation 
between crime type and PdDistrict and so PdDistrict is not a useful predictor variable and we can analyse and model
independently for each district. This also reduces the size of the dataset and enables logistics procedure to run
in a finite amount of time and memory*/
proc freq data=project.train;
table Category * PdDistrict / norow nocol nopercent;
run;

/*Modeling and prediction for MISSION District*/
title bold color=black H=3 'Analysis for PdDistrict - MISSION';

%CreatePredictorVars(project.train, project.train_MISSION, "MISSION", &MISSION_PD_X, &MISSION_PD_Y);
%CreateClusterVar(project.train_MISSION);
%CreateSampleTrain(project.train_MISSION, project.sampleTrain_MISSION, 10000);
%TrainModel(project.sampleTrain_MISSION, project.trainedModel_MISSION, project.predictions_MISSION);

%CreatePredictorVars(project.test, project.test_MISSION, "MISSION", &MISSION_PD_X, &MISSION_PD_Y);
%AssignClusterTest(project.test_MISSION)
%ScoreTest(project.trainedModel_MISSION, project.test_MISSION, project.scoredTest_MISSION);
title;

/*Modeling and prediction for SOUTHERN District*/
title bold color=black H=3 'Analysis for PdDistrict - SOUTHERN';

%CreatePredictorVars(project.train, project.train_SOUTHERN, "SOUTHERN", &SOUTHERN_PD_X, &SOUTHERN_PD_Y);
%CreateClusterVar(project.train_SOUTHERN);
%CreateSampleTrain(project.train_SOUTHERN, project.sampleTrain_SOUTHERN, 10000);
%TrainModel(project.sampleTrain_SOUTHERN, project.trainedModel_SOUTHERN, project.predictions_SOUTHERN);

%CreatePredictorVars(project.test, project.test_SOUTHERN, "SOUTHERN", &SOUTHERN_PD_X, &SOUTHERN_PD_Y);
%AssignClusterTest(project.test_SOUTHERN)
%ScoreTest(project.trainedModel_SOUTHERN, project.test_SOUTHERN, project.scoredTest_SOUTHERN);
title;
