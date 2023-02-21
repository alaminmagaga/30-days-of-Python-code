import re
from collections import Counter

def clean_text(text):
    # Remove all non-alphanumeric characters and convert to lowercase
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).lower()
    # Remove extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def most_frequent_words(text):
    # Split the text into words
    words = text.split()
    # Count the frequency of each word
    word_counts = Counter(words)
    # Return the three most common words
    return word_counts.most_common(3)

# Define the input string
sentence = '''%I $am@% a %tea@cher%, &and& I lo%#ve %tea@ching%;. There $is nothing; &as& mo@re rewarding as educa@ting &and& @emp%o@wering peo@ple. ;I found tea@ching m%o@re interesting tha@n any other %jo@bs. %Do@es thi%s mo@tivate yo@u to be a tea@cher!?'''

# Clean the text
cleaned_text = clean_text(sentence)
print("Cleaned text: ", cleaned_text)

# Count the three most frequent words
frequent_words = most_frequent_words(cleaned_text)
print("Most frequent words: ", frequent_words)
