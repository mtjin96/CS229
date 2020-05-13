from nltk.corpus import reuters
 
print(reuters.fileids())         # The list of file names inside the corpus
print(len(reuters.fileids()))            # Number of files in the corpus = 10788
 
# Print the categories associated with a file
print(reuters.categories('training/999'))      # [u'interest', u'money-fx']
 
# Print the contents of the file
print(reuters.raw('test/14829'))