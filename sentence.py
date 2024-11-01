# Import necessary libraries
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Define stemming function
content=input("Enter the title to be converted a stemmed word:")
ps = PorterStemmer()

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content


print(stemming(content))