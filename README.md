# Stack Overflow Questions with Quality Rating

## Task :pushpin:
Stack Overflow questions from 2016-2020 should be classified to one of the categories:
+ HQ: High-quality posts without a single edit.
+ LQ_EDIT: Low-quality posts with a negative score, and multiple community edits. However, they still remain open after those changes.
+ LQ_CLOSE: Low-quality posts that were closed by the community without a single edit.

<br/>

## Dataset :clipboard:
[Link to the dataset.](https://drive.google.com/drive/folders/1WyicJDvV0_d9y32WE9bRiVU95X1kOLwY?usp=share_link)

body - question body

tags - tags that the question refers to

title - name of the second company

target - category

<br/>

Example:
| body | tags | title |  target |
|----------------|:---------:|----------------:|----------------:|
| While converting the data frame to HTML, Date is getting converted to a number. \r\n\r\nHow to keep it date only? | `<html><r><dataframe>` | R Studio: Date is getting converted to number, while making html of datafrane | 2 |
| `<p>I'm setting a var using \n<code>set TEST_VAR=5</code> \nand then I'm compiling a C code. \nError found during compilation is TEST_VAR is an undeclared variable.</p>\n` | `<c><windows><batch-file>` | Environment variable set in batch file cannot be accessed in the C code compiled by the file | 1 |

<br/>

## EDA :chart_with_upwards_trend:

- Frequently encountered tags
<br/>
![Frequently encountered tags](/files/tag1.png)

- Minimally encountered tags
<br/>
![Minimally encountered tags](/files/tag2.png?raw=true)

- Distribution of targets
<br/>
![Distribution of targets](/files/target.png?raw=true)


## Preprocessing :scissors:

- removing words of length less than 4
- removing punctuation marks
- lowercase conversion
- removing stop words
- lemmatisation

<br/>

## Models :package:

### Sequential model(Keras) :white_circle:


____________________________________________________
 Layer (type)                Output Shape              Param #   
_________________________________________________________________
 embedding_2 (Embedding)     (None, 70, 300)           41575800  
                                                                 
 spatial_dropout1d_2 (Spatia  (None, 70, 300)          0         
 lDropout1D)                                                     
                                                                 
 lstm_2 (LSTM)               (None, 100)               160400    
                                                                 
 dense_6 (Dense)             (None, 1024)              103424    
                                                                 
 dropout_4 (Dropout)         (None, 1024)              0         
                                                                 
 dense_7 (Dense)             (None, 1024)              1049600   
                                                                 
 dropout_5 (Dropout)         (None, 1024)              0         
                                                                 
 dense_8 (Dense)             (None, 3)                 3075      
                                                                 
 activation_2 (Activation)   (None, 3)                 0  
 _________________________________________________________________

Metrics:

| batch_size  | optimizer  | accuracy |
|----------------|:---------:|----------------:|
| 512 | adam | 0.9 |
| 32  | sgd | 0.85 |
| 256  | adam | 0.9 |

<br/>


### fastText + SVM :white_circle:
Metrics:

| C | kernel  | accuracy |
|----------------|:---------:|----------------:|
| 1 | rbf | 0.88 |

<br/>

Metrics are based on a comparison of target.

## Performance :computer: 

CPU: Intel i5-10210U CPU @ 1.60GHz


Speed of classification using sequential model: 1090/1 sec

Speed of classification using fastText + SVM: 266,6/1 sec

<br/>

## Usage :information_desk_person:

Before using the project, you need to install the project dependencies:


```
pip install -r requirements.txt 
```

You can also test the project using the terminal.
<br/>

Sequential model:
```
python train.py --m lstm 
```
<br/>

fastText + SVM:
```
python train.py --m svm 
```

