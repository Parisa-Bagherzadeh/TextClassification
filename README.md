# Text Classification  
This text classification program utilizes GloVe (Global Vectors for Word Representation) embeddings to convert textual data into numerical vectors, enabling the training and evaluation of machine learning models for text classification tasks. GloVe embeddings capture semantic relationships between words, providing a meaningful representation for natural language processing tasks.  

## Getting Started 
1- clone the repository :  
```
git clone  https://github.com/Parisa-Bagherzadeh/TextClassification.git
cd TextClassification
```  
2 - Install the required dependencies :  
```
pip install -r requirements.txt
```  
3 - Download GloVe pre-trained embeddings. You can use the GloVe website (https://nlp.stanford.edu/projects/glove/) or the following command:  
```
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip -d embeddings/

```  
Choose the appropriate GloVe model based on your requirements (e.g., glove.6B.100d.txt for 100-dimensional embeddings). 

4 - Orgnize your data  
Place your training and testing datasets in the dataset/ directory. Ensure that your datasets are formatted appropriately, with one column for text data and another for corresponding labels.  

## Usage  
Run the text classification program:  
```
python text_classification.py --dimension Dimension_Of_Feature_Vector  --sentence Test_Sentence
```
The program will preprocess the data, load the GloVe embeddings, train a text classification model, and evaluate its performance on the testing dataset.  

#### Also I have test the model with/without dropout layer. Inference time will reduced when using dropout layer but the loss of both train and test data will increases and accuracy of train and test data will decreases

### Result without Dropout layer
<table>
    <tr>
        <td>Featue Vector Dimension</td>
        <td>Train Loss</td>
        <td>Train Accuracy</td>
        <td>Test Loss</td>
        <td>Test Accuracy</td>
        <td>Inference Time</td>
    </tr>
    <tr>
        <td>50d</td>
        <td>0.55</td>
        <td>0.85</td>
        <td>0.60</td>
        <td>0.85</td>
        <td>0.08s</td>
    </tr>    
    </tr>
        <td>100d</td>
        <td>0.39</td>
        <td>0.93</td>
        <td>0.55</td>
        <td>0.85</td>
        <td>0.09s</td>
    </tr>
    <tr>
        <td>200d</td>
        <td>0.20</td>
        <td>0.96</td>
        <td>0.44</td>
        <td>0.82</td>
        <td>0.10s</td>
    </tr>
    <tr>
        <td>300d</td>
        <td>0.12</td>
        <td>0.99</td>
        <td>0.41</td>
        <td>0.87</td>
        <td>0.10s</td>
    </tr>
   
</table>

<br>

### Result with Dropout layer
<table>
    <tr>
        <td>Featue Vector Dimension</td>
        <td>Train Loss</td>
        <td>Train Accuracy</td>
        <td>Test Loss</td>
        <td>Test Accuracy</td>
        <td>Inference Time</td>
    </tr>
    <tr>
        <td>50d</td>
        <td>0.71</td>
        <td>0.76</td>
        <td>0.69</td>
        <td>0.76</td>
        <td>0.09s</td>
    </tr>    
    </tr>
        <td>100d</td>
        <td>0.51</td>
        <td>0.84</td>
        <td>0.62</td>
        <td>0.83</td>
        <td>0.09s</td>
    </tr>
    <tr>
        <td>200d</td>
        <td>0.27</td>
        <td>0.93</td>
        <td>0.48</td>
        <td>0.78</td>
        <td>0.09s</td>
    </tr>
    <tr>
        <td>300d</td>
        <td>0.18</td>
        <td>0.98</td>
        <td>0.41</td>
        <td>0.89</td>
        <td>0.09s</td>
    </tr>
   
</table>