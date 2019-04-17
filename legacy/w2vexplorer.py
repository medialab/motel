#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys
reload(sys) 
sys.setdefaultencoding("utf-8")
#if not 'jean':
import matplotlib
matplotlib.use('Agg')
from sqlite3 import *
sys.path.append("../pylibrary")
from path import *
import fonctions
import re

import sqlite3
import shutil
#sys.path.append("../../bookdata/word2vec/")
import operator
import nltk
from graphlib import *
import codecs

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import nltk
from nltk.collocations import *
import gensim, logging
from gensim.models.doc2vec import LabeledSentence
from gensim.models.doc2vec import logger, FAST_VERSION
from operator import itemgetter
import scipy
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import seaborn as sns
import sklearn
import pickle
import math
import numpy as np
from sklearn.metrics import pairwise
from sklearn.cluster import DBSCAN
#from sklearn.cluster import OPTICS

from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import hdbscan

import networkx as nx
from networkx.readwrite import json_graph
import json
from gensim.models import Phrases
import itertools
import urllib2
try:
	print 'user_parameters',user_parameters
except:
	user_parameters=''
parameters_user=fonctions.load_parameters(user_parameters)
print 'parameters_user',parameters_user
print "FAST_VERSION",FAST_VERSION
import logging, pprint

##################################################################################
##################################################################################
##################################################################################
################################# FUNCTIONS ######################################
##################################################################################
##################################################################################
##################################################################################
#####FUNC
def dict_factory(cursor, row):
	d = {}
	for idx, col in enumerate(cursor.description):
		d[col[0]] = row[idx]
	return d

def dumpingin(data,datastr,resultpath=''):
	p = pickle.Pickler(open(os.path.join(pklpath,datastr+'.pkl'), 'wb')) 
	p.fast = True 
	p.dump(data) 


def dumpingout(datastr):
	print 'trying to retrieve ',datastr
	pkl_file = open(os.path.join(pklpath,datastr+'.pkl'), 'rb')
	data = pickle.load(pkl_file)
	print 'successfully retrieved ',datastr
	return data


def build_matrix(model,ensemble,samplesize=None):#possible to sample to build a smaller matrix
	arraylist=[]
	if samplesize==None:
		samplesize=len(ensemble)
	else:
		samplesize=min(samplesize,len(ensemble))
	if samplesize<len(ensemble):
		ensemblesampled = random.sample(ensemble, samplesize)
	else:
		ensemblesampled=ensemble
	for id in ensemblesampled:#idq_2000[:50]:
		#print np.array(model.docvecs[id]).shape
		try:
			arraylist.append(model.docvecs[id])
		except:
			try:
				arraylist.append(model[id])
			except:
				wordlist=contentall[id[3:]]
				#if len(wordlist)>0:
				arraylist.append(infer_vectors(wordlist,model))


			
	matrix= np.array(arraylist)
	try:
		matrix=matrix[:,0,:]
	except:
		pass
	return matrix


# In[8]:

def moyenne_vectors(model,ids):
	moyenne=model.docvecs[ids[0]]
	hh=0
	if len(ids)>1:
		for id in ids[1:]:
			try:
				moyenne+=model.docvecs[id]
				hh+=1
			except:
				pass
	moyenne = moyenne / float(hh)
	return moyenne

def compute_similarity(v1,v2):
	return np.dot(v1, v2)/(np.linalg.norm(v1)* np.linalg.norm(v2))


# In[8]:




# In[10]:

def infer_vectors(wordlist,model,mode='tfidf'):
	vecs=[]
	for word in wordlist:
		vecs.append(counterlog[word]*model[word].reshape((1,modeldimensions)))
	try:
		vecs = np.concatenate(vecs)
	except:
		vecs=numpy.zeros((1,modeldimensions))
	#vecs=np.array(vecs, dtype='float') #TSNE expects float type values
	return vecs.sum(axis=0).reshape((1,modeldimensions))

def compute_similarity(v1,v2):
	cosine_similarity = np.dot(v1, v2)/(np.linalg.norm(v1)* np.linalg.norm(v2))
	return cosine_similarity

def get_vectors(words,model):#DD-75.MOE@DIRRECTE.gouv.fr
	vecs=[]
	for word in words:
		try:
			vecs.append(model[word].reshape((1,modeldimensions)))
		except:
			try:
				vecs.append(model.docvecs[word].reshape((1,modeldimensions)))
			except:
				wordlist=contentall[word[3:]]
				#if len(wordlist)>0:
				vecs.append(infer_vectors(wordlist,model).reshape((1,modeldimensions)))
			  
	vecs = np.concatenate(vecs)
	vecs=np.array(vecs, dtype='float') #TSNE expects float type values
	return vecs
	
def compute_pos(model,sorted_x,topn,name='',show=True):
	wordsflat=map(lambda x:x[0],sorted_x[:topn])
	#print wordsflat
	if len(wordsflat[0])==1:
		words=sorted_x[:topn]
	else:
		words=wordsflat
	try:
		reducedvecsname='reduced_vecs2'+name+str(topn)
		print 'trying to retrieve...',reducedvecsname
		reduced_vecs2=dumpingout(reducedvecsname)
		vicsd2=dumpingout('vicsd2'+name+str(topn))
		print 'successing...'
	except:
		print "recomputing the 2d projection"
		print len(words)
		vics2=get_vectors(words,model)
		vicsd2=sklearn.metrics.pairwise.pairwise_distances(vics2, vics2,'cosine')
		vicsd2 = vicsd2- vicsd2.min()#small bug in cosine distance computation
		reduced_vecs2=manifoldreduct(vicsd2,'TSNE','precomputed')
		dumpingin(reduced_vecs2,'reduced_vecs2'+name+str(topn))
		dumpingin(vicsd2,'vicsd2'+name+str(topn))
	plotit(reduced_vecs2,words,{},name='test'+name+'nbw'+str(topn),neighdict={},show=show)
	return reduced_vecs2,vicsd2
		
#
# def plottopwords(model,sorted_x,topn,name='',neighdict={},show=True,distinctafter=100000):
# 	try:
# 		h=sorted_x[0][2]
# 		words=sorted_x[:topn]
# 	except:
# 		words=map(lambda x:x[0],sorted_x[:topn])
# 	try:
# 		print 'trying...'
# 		reduced_vecs2=dumpingout('sreduced_vecs2'+name+str(len(words)))
#
# 		print 'succesing...'
# 	except:
# 		print len(words)
# 		vics2=get_vectors(words,model)
# 		vicsd2=sklearn.metrics.pairwise.pairwise_distances(vics2, vics2,'cosine')
# 		vicsd2 = vicsd2- vicsd2.min()#small bug in cosine distance computation
# 		reduced_vecs2=manifoldreduct(vicsd2,'TSNE','precomputed')
# 		dumpingin(reduced_vecs2,'reduced_vecs2'+name+str(len(words)))
#
#
# 	print 'test'+name+'nbw'+str(len(words))
# 	plotit(reduced_vecs2,words,{},name='test'+name+'nbw'+str(len(words)),neighdict=neighdict,show=show,distinctafter=distinctafter)

	
def plotit(reduced_vecs,words,dcounter,neighdict={},typex='point',name='',moyenne={},poids={},show=False,distinctafter =100000):
	hexbin=False
	neighdict_inv = {}
	neighdictlist=neighdict.keys()
	for x,voisins in neighdict.iteritems():
		for y in voisins:
			#print 'y',y
			neighdict_inv.setdefault(y,[]).append(x)
	print " len(neighdict)", len(neighdict)
	current_palette = sb.color_palette("Set2", len(neighdict))
	size=np.log(len(reduced_vecs))*2
	fig=plt.figure(figsize=(size,size))
	print "len(words)",len(words)
	for i,w in enumerate(words):
		#print 'w',w
		if 1:
			if 'pseudo_' in w:
				plt.text(reduced_vecs[i,0], reduced_vecs[i,1],w.decode('utf-8'),fontsize=10,color='b')
				plt.plot(reduced_vecs[i,0], reduced_vecs[i,1], marker='o', color='b', markersize=np.log(counter.get(w,400)),alpha=.5)
			else:
				if w in neighdict_inv:
					for h in range(len(neighdict_inv[w])):
						plt.plot(reduced_vecs[i,0], reduced_vecs[i,1], marker='o', color=current_palette[neighdictlist.index(neighdict_inv[w][h])], markersize=2,alpha=.5)
						plt.text(reduced_vecs[i,0], reduced_vecs[i,1],w.decode('utf-8'),fontsize=4)
					#for meta in neighdict_inv[w]:
					#	if meta[0]==' ':
					#		plt.text(reduced_vecs[i,0], reduced_vecs[i,1], meta[:2],fontsize=4)
					#	else:
					#		plt.text(reduced_vecs[i,0], reduced_vecs[i,1], meta[:1],fontsize=4)
				else:
					if i>distinctafter:
						plt.plot(reduced_vecs[i,0], reduced_vecs[i,1], marker='o', color='r', markersize=2,alpha=.1)
					else:
						plt.plot(reduced_vecs[i,0], reduced_vecs[i,1], marker='o', color='b', markersize=4,alpha=.1)
					if show:
							plt.text(reduced_vecs[i,0], reduced_vecs[i,1], w.decode('utf-8'),fontsize=1)
	print 'map ', os.path.join(result_path,name+'.pdf'), ' created'
	plt.savefig(os.path.join(result_path,name+'.pdf'))
#####PARAMETERS

#DATA PROCESSING OPTIONS

show=True
log_every=10000

data_source = parameters_user.get('corpus_file','')
pklpath=os.path.join('/'.join(parameters_user.get('corpus_file','').split('/')[:-1]),'pkl')
try:
	os.mkdir(pklpath)
except:
	pass
result_path=parameters_user.get('result_path','')
result_path0=result_path[:]
fonctions.progress(result_path0,0)

print 'avant param'
try:
	print 'trying'
	tablename=parameters_user.get('tablename','ISIABSTRACT')
	try:
		min_count=int(parameters_user.get('min_count',5))
	except:
		min_count=5
	cut_sentences=parameters_user.get('cut_sentences',False)
	try:
		bigram_threshold=int(parameters_user.get('bigram_threshold',100))
	except:
		bigram_threshold=5
	try:
		window_size=int(parameters_user.get('bigram_threshold',100))
	except:
		window_size=10
	try:
		modeldimensions=int(parameters_user.get('modeldimensions',100))
	except:
		modeldimensions=100
	bigramactivated=parameters_user.get('bigramactivated',False)
	trigramactivated=parameters_user.get('trigramactivated',False)
	try:
		min_len=int(parameters_user.get('min_len',2))
	except:
		min_len=2
	remove_accent=parameters_user.get('remove_accent',False)
	lemmatization=parameters_user.get('lemmatization',False)
	try:
		iteration=int(parameters_user.get('iteration',5))
	except:
		iteration=5
	try:
		max_vocab_size=int(parameters_user.get('max_vocab_size',None))
	except:
		max_vocab_size=None
	try:
		sizemap=int(parameters_user.get('sizemap',1000))
	except:
		sizemap=1000
	if sizemap>5000:
		if str(sizemap)[-1]=='9':
			pass
		else:
			sizemap=5000
	clustering_space=parameters_user.get('clustering_space','t-SNE')
	
	try:
		min_cluster_size=int(parameters_user.get('min_cluster_size',10))
	except:
		min_cluster_size=10
	try:
		min_samples=int(parameters_user.get('min_samples',1))
	except:
		min_samples=1
	force_recompute =parameters_user.get('force_recompute',False)
	gensimmodel='Word2Vec'
	try:
		nbworkers=int(parameters_user.get('nbworkers',1))
		if nbworkers==999:
			nbworkers=11
		elif nbworkers==9999:
			gensimmodel='Doc2Vec'
			if 'cointet' in os.getcwd():
				nbworkers=3
			else:
				nbworkers=11
		else:
			if nbworkers>3:
				nbworkers=3
	except:
		nbworkers=1
	print 'learning with ',nbworkers,' workers'
except:
	tablename='ISIABSTRACT'
	cut_sentences=True
	window_size=10
	modeldimensions=100
	
	bigramactivated=True
	trigramactivated=False
	min_len=2
	remove_accent=False
	lemmatization=True
	iteration=20
	max_vocab_size=None
	#DATA MAPPING OPTIONS
	sizemap=2261
	clustering_space='t-SNE'

	min_cluster_size=int(math.log(sizemap))
	print 'min_cluster_size',min_cluster_size
	min_cluster_size=10
	min_samples=1
modelparameters='_'.join(map(lambda x:str(x),[data_source.split('/')[-1].split('.db')[0],tablename,min_count,cut_sentences,bigram_threshold,window_size,modeldimensions,bigramactivated,trigramactivated,min_len,remove_accent,lemmatization,iteration,max_vocab_size]))
vizparameters='_'.join(map(lambda x:str(x),[sizemap,clustering_space,min_cluster_size]))
#####CODE
print 'trying to open', data_source



##################################################################################
##################################################################################
##################################################################################
############################# DATA PROCESSING ####################################
##################################################################################
##################################################################################
##################################################################################
	
SPLIT_SENTENCES = re.compile(u"[.!?:]\s+")  # split sentences on these characters
def process_message(message):
	message = gensim.utils.to_unicode(message, 'utf-8').strip()
	contents=[]
	if cut_sentences:
		for sentence in SPLIT_SENTENCES.split(message):
			#content=list(gensim.utils.tokenize(sentence, lower=True))
			if lemmatization:
				content=gensim.utils.lemmatize(sentence)
				content=list(gensim.utils.simple_preprocess(' '.join(map(lambda x: x.split('/')[0],content)), deacc=remove_accent, min_len=min_len, max_len=15))
			else:
				content=list(gensim.utils.simple_preprocess(sentence, deacc=remove_accent, min_len=min_len, max_len=15))
			contents.append(content)
	else:
		if lemmatization:
			content=gensim.utils.lemmatize(message)
			contents=[list(gensim.utils.simple_preprocess(' '.join(map(lambda x: x.split('/')[0],content)), deacc=remove_accent, min_len=min_len, max_len=15))]
		else:
			contents=[list(gensim.utils.simple_preprocess(message, deacc=remove_accent, min_len=min_len, max_len=15))]
			#contents=[message.split()]
	return contents

#print process_message(message)
dbname=data_source
connection = sqlite3.connect(dbname)
connection.row_factory = dict_factory
cursor = connection.cursor()
bigram = Phrases()

def iter_rawcontent(iterator, log_every=log_every):	
	extracted = 0
	for row in  iterator:#extracted<100:
		content = row['data']
		if extracted % log_every == 0:
			print "extracting content" +  str(extracted)	
		for sentence in  process_message(content):
			yield(sentence)
		extracted += 1

def iter_rawcontentd2v(iterator, log_every=log_every):	
	extracted = 0
	for row in  iterator:#extracted<100:
		content = row['data']
		if extracted % log_every == 0:
			print "extracting content" +  str(extracted)	
		for sentence in  process_message(content):
			yield(sentence,extracted)
		extracted += 1

def iter_phrases(iterator, bigram,log_every=log_every):
	extracted = 0
	for content in iterator:
		#print "content",content
		if extracted % log_every == 0:
			print "extracting content" +  str(extracted)
		yield bigram[content]
		extracted += 1

class Myphrases(object):
	def __init__(self, iterator, bigram):
		self.iterator = iterator
		self.bigram = bigram
		print 'go'
	def __iter__(self):
		for content in self.iterator:
			try:
				yield self.bigram[content]
			except:
				yield content

class Myphrasesd2v(object):
	def __init__(self, iterator, bigram):
		self.iterator = iterator
		self.bigram = bigram
		print 'go'
	def __iter__(self):
		for content in self.iterator:
			try:
				yield self.bigram[content]
			except:
				yield LabeledSentence(content[0],[content[1]])


##################################################################################
##################################################################################
############################# MAPPING CODE #######################################
##################################################################################
##################################################################################
##################################################################################
#################################
###logging user parameters#######
#################################
logging.basicConfig(filename=os.path.join(result_path,'.user.log'), filemode='w', level=logging.DEBUG,format='%(asctime)s %(levelname)s : %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
logging.info('Script W2VExplorer Started')
yamlfile = 'w2vexplorer.yaml'
if 'cointet' in os.getcwd():
	yamlfile = '/Users/jpcointet//Desktop/cortext-methods/w2vexplorer/'+yamlfile
parameterslog=fonctions.getuserparameters(yamlfile,parameters_user)
logger = logging.getLogger()
logging.info(parameterslog)
logger.handlers[0].flush()
fonctions.progress(result_path0,2)


##################################
### end logging user parameters###
##################################


print "mapping starting"
fname=gensimmodel[:3]+modelparameters+'.mod'
name=modelparameters+'_'+vizparameters
try:
	if force_recompute:		
		djqs
	X_r=dumpingout('reduced_vecs2'+name+str(sizemap))
	vicsd2=dumpingout('vicsd2'+name+str(sizemap))
	word2map=dumpingout('word2map'+name+str(sizemap))
	modelall = gensim.models.word2vec.Word2Vec.load(os.path.join(pklpath,fname))
	print "successfully retrieved old positions"
	fonctions.progress(result_path0,78)
except:	
	print "***recomputing positions"
	logging.info('Computing positions')
	try:
		if force_recompute:
			djqs
		if gensimmodel=='Doc2Vec':
			modelall = gensim.models.Doc2Vec.load(os.path.join(pklpath,fname))
			logging.info('Loading extisting model')
			
		else:
			modelall = gensim.models.word2vec.Word2Vec.load(os.path.join(pklpath,fname))
			logging.info('Loading extisting model')
		print "model was retrieved from a former training"
		logging.info('Extisting model loaded')
	except:
		print "model should be computed"
		fonctions.progress(result_path0,5)	
		NN=fonctions.count_rows(dbname,tablename)
		logging.info('Model shoud be recomputed')
		iterator,iterator2=itertools.tee(cursor.execute("select data from "+tablename))# +' limit 10000'))
		if gensimmodel=='Doc2Vec':
			iterateur=iter_rawcontentd2v(iterator)
		else:
			iterateur=iter_rawcontent(iterator)
		if bigramactivated:
			bigram = Phrases(iterateur, min_count=min_count, threshold=bigram_threshold)
			iterateurraw0=iter_rawcontent(iterator2)
			if trigramactivated:
				iterateurph0,iterateurph2=itertools.tee(iter_phrases(iterateurraw0,bigram))
				bigram2 = Phrases(iterateurph2, min_count=min_count, threshold=bigram_threshold)
				iterateurph3= Myphrases(iterateurph0,bigram2)
			else:
				iterateurph3= Myphrases(iterateurraw0,bigram)
		else:
			if gensimmodel=='Doc2Vec':
				iterateurph3= Myphrasesd2v(iterateur,{})
			else:
				iterateurph3= Myphrases(iterateur,{})
			#iterateurph4=MySentences(iterateurph3)
		print "now learning"
		logging.info('Now learning the model')
		if gensimmodel=='Doc2Vec':
			print 'gensimmodel',gensimmodel
			modelall = gensim.models.Doc2Vec(size=modeldimensions, window=window_size, workers=int(nbworkers),dbow_words =1,dm=0,min_count=min_count,iter=iteration,max_vocab_size=max_vocab_size)		
		else:	
			print 'gensimmodel',gensimmodel
			modelall = gensim.models.word2vec.Word2Vec(size=modeldimensions, window=window_size, workers=int(nbworkers),min_count=min_count,iter=iteration,max_vocab_size=max_vocab_size)		
		print modeldimensions, window_size, min_count,iteration,max_vocab_size
		
		iterateurph3=list(iterateurph3)
		print "Number of items to be learnt: ",len(iterateurph3)
		logging.info(str(len(iterateurph3)) + ' entries used for learning')
		logging.getLogger().setLevel(logging.CRITICAL)
		modelall.build_vocab(iterateurph3)
		logging.getLogger().setLevel(logging.DEBUG)
		print 'Vocab size: ',len(modelall.wv.vocab)
		fonctions.progress(result_path0,50)
		logging.info('Vocabulary size is '+str(len(modelall.wv.vocab)))
		logging.getLogger().setLevel(logging.CRITICAL)
		modelall.train(iterateurph3,total_examples=NN,epochs=modelall.iter)
		logging.getLogger().setLevel(logging.DEBUG)
		logging.info(' Model was trained')
		modelall.save(os.path.join(pklpath,fname))
		logging.info(' Model was saved')		
		fonctions.progress(result_path0,55)
		
	print fname, 'model saved'
	if parameters_user.get('mapmodel',True):
		contentall={}
		vocaball=modelall.wv.vocab
		counter={}
		counterlog={}
		for x in modelall.wv.vocab:
			counter[x]=modelall.wv.vocab[x].count
			counterlog[x]=math.log(100000./float(modelall.wv.vocab[x].count))
		words_sorted_x = sorted(counter.items(), key=operator.itemgetter(1),reverse=True)
		print words_sorted_x[:10]
		print len(words_sorted_x)


		word2map=map(lambda x:x[0],words_sorted_x[:sizemap])
		word2map=[x for x in word2map if x!="source" and x!="target"]
		matq2000all=build_matrix(modelall,word2map,samplesize=8947342)
		logging.info(' Distance Matrix between words computed')		
		dumpingin(word2map,'word2map'+name+str(sizemap))
		print "word2map",word2map
		#sizemap=len(word2map)
		fonctions.progress(result_path0,60)
		logging.info('Computing word positions with t-SNE')		
		try:
			X_r,vicsd2=compute_pos(modelall,word2map,sizemap,name=name)
		except:
			print 'Vocabulary size is too small, t-SNE projection cannot be computed'
			logging.debug('Vocabulary size is too small, t-SNE projection cannot be computed')
			logging.info('Word positions computed')		
	

# In[165]:
fonctions.progress(result_path0,80)
if parameters_user.get('mapmodel',True):
	if clustering_space=='t-SNE':
		clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples,metric='euclidean')
		X=X_r
		X = StandardScaler().fit_transform(X_r)
		logging.info('Clusters computed in t-SNE space')		
	
	else:
		clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,min_samples=min_samples,metric='precomputed')
		X =vicsd2
		logging.info('Clusters computed in W2Vec space')		
fonctions.progress(result_path0,90)
	
#X = StandardScaler().fit_transform(matq2000all)
#X=X.astype("double")
#
#print X[0]
if parameters_user.get('mapmodel',True):

	cluster_labels = clusterer.fit_predict(X)
	unique_labels=list(set(cluster_labels))
	dico_cluster={}
	cluster_label={}
	for word,label_id,word_centrality in zip(word2map,cluster_labels,clusterer.probabilities_):
		dico_cluster.setdefault(label_id,[]).append((word,word_centrality))
	for label_id, coupleslist in dico_cluster.iteritems():
		bestwords= sorted(coupleslist, key=operator.itemgetter(1),reverse=True)
		cluster_label[label_id]= bestwords[0][0]
		try:
			cluster_label[label_id]= cluster_label[label_id] + ' & '+ bestwords[1][0]
		except:
			pass


	for label_id,words in dico_cluster.iteritems():
		print '***'
		print label_id,cluster_label[label_id],'; '.join(map(lambda x:x[0],words))

		#dico_cluster.setdefault(label_id,[]).append(word)

	print len(unique_labels), ' different clusters'
	colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
	#colors=sns.color_palette("Set2", )
	#colors=sns.color_palette("Set1", n_colors=len(unique_labels), desat=.5)
	#colors=sns.color_palette( n_colors=len(unique_labels))
	#random.shuffle(colors)
	clusize=[]


	# for k, col in zip(unique_labels, colors)[:]:
	# 	if k == -1:
	# 		# Black used for noise.
	# 		col = 'k'
	#
	# 	class_member_mask = (cluster_labels == k)
	# 	xy = X_r[class_member_mask]
	# 	if k!=-1:
	# 		clusize.append(len(xy))
	# 	else:
	# 		print "that many words not categorized",len(xy)
	# 	plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
	# 			 markeredgecolor='k', markersize=1)
	# #listefinale=map(lambda x: x[0],words_sorted_x[:sizemap])
	# for i,word in enumerate(word2map):
	# 	try:
	# 		clab=cluster_labels[i]
	# 		plt.text(X_r[i, 0], X_r[i, 1],str(clab)+'_'+word.decode('utf-8'),fontsize=2,color=colors[cluster_labels[i]])
	# 		#print X_r[i, 0], X_r[i, 1],str(clab)+'_'+unicode(word,'utf-8')
	# 	except:
	# 		pass
	# plt.savefig('hdbscancluster.pdf')
	#plt.figure()
	#hist(clusize,20)



	#import sys
	#reload(sys)
	#sys.setdefaultencoding("utf-8")
	colorsrgb=[]
	for color in colors:
		rbgtriplet=[]
		for value in color[:3]:
			rbgtriplet.append(int(value * 255))
		colorsrgb.append(rbgtriplet)
	print colorsrgb
	G=nx.Graph()

	topwordslist = word2map#map(lambda x: unicode(x[0],'utf-8'),words_sorted_x[:sizemap])
	#G.add_nodes_from(list(set(([x.decode('utf-8') for x in topwordslist]))))
	#G.add_nodes_from(topwordslist)
	i=-1
	minsize=10000
	for node in topwordslist:
		minsize=min(minsize,math.log(modelall.wv.vocab[node].count))
	print "avant",len(topwordslist)
	#print cluster_labels
	for node in topwordslist:# [x for x in topwordslist if x!='source' and x!='target']:#G.nodes():#topwordslist:
		nodeinfo={}
		i+=1
		nodeori=node[:]
		#node=unicode(node,'utf-8')
		G.add_node(node)
		#print "node ", node, " added",
		if 1:
			G.node[node]['attributes']={}
			G.node[node]['attributes']['weight'] = 1
			G.node[node]['label'] = node
			#G.node[node]['position']={}
			G.node[node]['x']=X_r[i,0]
			G.node[node]['y']=X_r[i,1]
			#print node,X_r[i,0],X_r[i,1]
			G.node[node]['id']=node
			#G.node[node]['viz']['position']['z']=0
			G.node[node]['size']=math.log(modelall.wv.vocab[nodeori].count)-minsize+1
			#G.node[node]['viz']['color']={}
			#print "cluster_labels[i]",cluster_labels[i],node
			#print colorsrgb[cluster_labels[i]]
			G.node[node]['color']='rgb'+str(tuple(colorsrgb[cluster_labels[i]])).replace(' ','')
			#G.node[node]['viz']['color']['r']=colorsrgb[cluster_labels[i]][0]
			#G.node[node]['viz']['color']['g']=colorsrgb[cluster_labels[i]][1]
			#G.node[node]['viz']['color']['b']=colorsrgb[cluster_labels[i]][2]
			G.node[node]['attributes']['cluster_label']=cluster_label[cluster_labels[i]]
			#G.node[node]['cluster_label']=int(cluster_labels[i])+1
		#print G.node[node]
		neigh=modelall.most_similar(nodeori,topn=100)
		for n in neigh:
			if n[0] in topwordslist:
				if n[1]>.5:
					G.add_edge(node,n[0],weight=n[1])
				#G.add_edge(n[0].decode('utf-8'),node,weight=n[1])



	#f = codecs.open("ordre1.csv", "r", "utf-8")

	logging.info('Final visualization produced')		
fonctions.progress(result_path0,95)
if parameters_user.get('mapmodel',True):

	print len(G.edges())
	print "len(G.nodes())",len(G.nodes())
	def save(G, fname):
		try:
			json.dump(dict(nodes=[G.node[n] for n in G.nodes()],
					   edges=[{"attributes":{},'source':u,'target':v, 'size':G.edge[u][v]['weight'], "color":"rgb(102,138,163)","id":str(i)} for i,(u,v) in enumerate(G.edges())]),
				  codecs.open(fname, 'w','utf-8'), indent=2)
		except:
			pass
			#json.dump(dict(nodes=[G.node[n] for n in G.nodes()],
			#			   edges=[{"attributes":{},'source':u,'target':v, "color":"rgb(102,138,163)","id":str(i)} for i,(u,v) in enumerate(G.edges())]),
			#		  codecs.open(fname, 'w','utf-8'), indent=2)
			
	try:
		os.mkdir(os.path.join(result_path,'data'))
	except:
		pass
	datafile=os.path.join(os.path.join(result_path,'data'),'W2V'+modelparameters+'_'+vizparameters+'.json')
	save(G,datafile)


	dataclu=os.path.join(os.path.join(result_path,'data'),'W2V'+modelparameters+'_'+vizparameters+'.csv')
	dataclu2=os.path.join(os.path.join(result_path,'data'),'W2V'+modelparameters+'_'+vizparameters+'eq.csv')
	print 'dataclu',dataclu
	feedcsv=open(dataclu,'w')
	feedcsv.write('stem\tmain form\tforms\n')
	feedcsv2=open(dataclu2,'w')
	feedcsv2.write('form\tequivalent form\n')
	for word in G.nodes():
		w = word.replace('_',' ')
		feedcsv.write(w+ '\t'+ w  +'\t'+ w + '\n')
		feedcsv2.write(w+ '\t'+G.node[word]['attributes']['cluster_label'] + '\n')
	feedcsv.close()
	feedcsv2.close()

	hashstring=result_path.split('/')[-2]
	print 'hashstring',hashstring
	url='https://assets.cortext.net/docs/'+hashstring+'/meta'
	print 'url',url
	datajson=json.loads(urllib2.urlopen(url).read())['files']
	print 'datajson',datajson
	dataurlassets = [datajson[x]['url'] for x in datajson if '/data' in datajson[x]['filename'] and '.json' in datajson[x]['filename']][0]
	print 'dataurlassets',dataurlassets

	config={
	  "type": "network",
	  "version": "1.0",
	  "data": "config.json",
	  "logo": {
	    "file": "images/logo.jpg",
	    "link": "",
	    "text": ""
	  },
	  "text": {
	    "more": "",
	    "intro": "~2000 words vectors spatialized in 2d using t-SNE and cluster with HDBScan",
	    "title": "SB Papers"
	  },
	  "legend": {
	    "edgeLabel": "",
	    "colorLabel": "",
	    "nodeLabel": ""
	  },
	  "features": {
	    "search": True,
	    "groupSelectorAttribute": "cluster_label",
	    "hoverBehavior": "dim"
	  },
	  "informationPanel": {
	    "groupByEdgeDirection": False,
	    "imageAttribute": False
	  },
	  "sigma": {
	    "drawingProperties": {
	      "defaultEdgeType": "curve",
	      "defaultHoverLabelBGColor": "#002147",
	      "defaultLabelBGColor": "#ddd",
	      "activeFontStyle": "bold",
	      "defaultLabelColor": "#000",
	      "labelThreshold": 4,
	      "defaultLabelHoverColor": "#fff",
	      "fontStyle": "bold",
	      "hoverFontStyle": "bold",
	      "defaultLabelSize": 20
	    },
	    "graphProperties": {
	      "maxEdgeSize": 0.01,
	      "minEdgeSize": 0.0,
	      "minNodeSize": 0.5,
	      "maxNodeSize": 5
	    },
	    "mouseProperties": {
	      "maxRatio": 40,
	      "minRatio": 0.75
	    }
	  }
	}
	#config['data']=datafile
	config['data']=dataurlassets
	config['text']['more']=str(parameters_user)[1:-1]
	config['text']['intro']= tablename
	config['text']['title']=str(data_source.split('/')[-1].split('.db')[0])
	print os.path.join(result_path,'W2V.viz')
	with open(os.path.join(result_path,'W2V.viz'), 'w') as outfile:
		json.dump(config, outfile,indent=4)
	
fonctions.progress(result_path0,100)	
logging.info('Script W2vexplorer ended successfully')		
fonctions.progress(result_path0,100)


