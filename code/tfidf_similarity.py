import json, os, traceback, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

class TfIdfSimilarity(object):

	def __init__(self):
		self.stop_words = list(stopwords.words('english'))
		self.lemmatizer = WordNetLemmatizer()


	def takeVerbNoun(self, text):
		return " ".join([tpl[0] for tpl in pos_tag(text.split()) if tpl[1] in ['NN','VB'] ])

	def word_token(self, tokens, lemma=False):
		tokens = str(tokens)
		tokens = re.sub(r"([\w].)([\~\!\@\#\$\%\^\&\*\(\)\-\+\[\]\{\}\/\"\'\:\;])([\s\w].)", "\\1 \\2 \\3", tokens)
		tokens = re.sub(r"\s+", " ", tokens)
		if lemma:
			return " ".join([self.lemmatizer.lemmatize(token, 'v') for token in word_tokenize(tokens.lower()) if token not in self.stop_words and token.isalpha()])
		else:
			return " ".join([token for token in word_tokenize(tokens.lower()) if token not in self.stop_words and token.isalpha()])

	def tfIdfVectorizer(self, jd, resume):
		cosine_similarities = ''
		try:
			# print("\n TF-TDF Vectorizer --- ")
			# print("\n JD 333--- ", jd, len(jd))
			# print("\n resume 333--- ",resume, len(resume))
			common_words = list(set(jd.split()) & set(resume.split()))
			# print("\n common words --- ", common_words, len(common_words))
			vectorizer = TfidfVectorizer(use_idf = True, sublinear_tf=True, lowercase = True) # sublinear_tf=True, analyzer='word', , ngram_range=(1,2)
			jd_vector = [jd]
			jd_vector = vectorizer.fit_transform(jd_vector)
			resume_vector = vectorizer.transform([resume])
			cosine_similarities = cosine_similarity(jd_vector, resume_vector).flatten()[0]
			# print("\n tfidf similarity --- ", cosine_similarities)
		except Exception as e:
			print("\n Error in tfIdfVectorizer() ",e, "\n ",traceback.format_exc())
		return cosine_similarities

	def cal_consine_similarities(self, jd, resume):
		jd = self.word_token(jd, True)
		resume = self.word_token(resume, True)
		resume = self.takeVerbNoun(resume)
		return self.tfIdfVectorizer(jd, resume)

if __name__ == '__main__':
	obj = TfIdfSimilarity()
	response = "The whole point of cheerleading is to show off their skills, so I’m sure they get paid a lot of money."
	context = "Do you know if professional cheerleaders make a lot of money?"
	evidence = "Cheerleading: Cheerleading originated in the United States with an estimated 1.5 million participants in all-star cheerleading."
	try:
		res_ctx_score = obj.cal_consine_similarities(response, context)
		res_evi_score = obj.cal_consine_similarities(response, evidence)
	except:
		res_ctx_score = 0
		res_evi_score = 0
	print(res_ctx_score, res_evi_score)
	