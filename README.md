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
![Frequently encountered tags](https://github.com/ritchann/stack-overflow-questions-quality/blob/main/files/tag1.png)

![alt text](https://github.com/[username]/[reponame]/blob/[branch]/files/tag1.png?raw=true)

## Preprocessing :scissors:

- removing words of length less than 4
- removing punctuation marks
- lowercase conversion
- removing stop words
- lemmatisation

<br/>

## Models :package:

### Search for similar names
1. TF-IDF, K-means clustering, Levenshtein distance

2. Word2vec, MiniBatchKMeans, Levenshtein distance

<br/>

## Performance :computer: 

CPU: Intel i5-10210U CPU @ 1.60GHz


To compare two values using the Levenshtein distance: 6550/1sec

Speed of processing a request for similar names(Word2vec, MiniBatchKMeans, Levenshtein distance): 0.47sec

<br/>

## Usage :information_desk_person:

You can open tutorial.ipynb to demonstrate the work of the project. Before using the project, you need to install the project dependencies:


```
pip install -r requirements.txt 
```

You can also test the project using the terminal.
<br/>

Levenshtein distance:
```
python train.py --m ld --name1 "Name 1" --name2 "Name 1" 
```
<br/>

TF-IDF, K-means clustering, Levenshtein distance:
```
python train.py --m tf --name1 "Name" 
```
<br/>

Word2vec, MiniBatchKMeans, Levenshtein distance:
```
python train.py --m w2 --name1 "Name"
```

