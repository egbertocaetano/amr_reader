'''
Created on 21 de ago de 2017

@author: wei
'''


import os
import subprocess
import re
from nltk.tokenize import sent_tokenize

class Preprocessor:

    def __init__(self):
        ROOT_PATH = os.getcwd()
        # self.SPLITTER_SENTENCE_PATH = "src/preprocessor/tools/text_spliter"
        self.SPLITTER_SENTENCE_PATH = "preprocessor/tools/text_spliter"
        self.SPLITTER_SENTENCE = os.path.join(ROOT_PATH, self.SPLITTER_SENTENCE_PATH)
#         self.SPLITTER_SENTENCE = "/home/forrest/workspace/LINE"
        self.max_sentence_len = 100

    def heuristic_sentence_splitting(self, raw_sent):
        if len(raw_sent) == 0:
            return []
        
        if len(raw_sent.split()) <= self.max_sentence_len:
            return [raw_sent]
  
        i = len(raw_sent) / 2
        j = int(i)
        k = int(i + 1)
        boundaries = [';', ':', '!', '?']
        
        results = []
        while j > 0 and k < len(raw_sent) - 1:
            if raw_sent[j] in boundaries:
                l_sent = raw_sent[ : j + 1]
                r_sent = raw_sent[j + 1 : ].strip()
                
                if len(l_sent.split()) > 1 and len(r_sent.split()) > 1:
                    results.extend(self.heuristic_sentence_splitting(l_sent))
                    results.extend(self.heuristic_sentence_splitting(r_sent))
                    return results
                else:
                    j -= 1
                    k += 1
            elif raw_sent[k] in boundaries:
                l_sent = raw_sent[ : k + 1]
                r_sent = raw_sent[k + 1 : ].strip()
                
                if len(l_sent.split()) > 1 and len(r_sent.split()) > 1:
                    results.extend(self.heuristic_sentence_splitting(l_sent))
                    results.extend(self.heuristic_sentence_splitting(r_sent))
                    return results
                else:
                    j -= 1
                    k += 1
            else:
                j -= 1
                k += 1
        
        if len(results) == 0:
            return [raw_sent]

                
    def raw_document_splitter(self, file_path):
        
        
        seg_sents = []
        cmd = 'perl %s/boundary.pl -d %s/HONORIFICS -i %s' % (self.SPLITTER_SENTENCE, self.SPLITTER_SENTENCE, os.path.abspath(file_path))
        
#         p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.PIPE, shell = True, universal_newlines=True)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, encoding='latin1')
        output, errdata = p.communicate()
        
        if len(errdata) == 0:
            
            raw_document = self.preprocess_punctuation(output)
            raw_document = raw_document.strip().split('\n\n')
            
            for raw_string in raw_document:
                raw_sentences = raw_string.split('\n')
                for raw_sent in raw_sentences:
                    if len(raw_sent.split()) > self.max_sentence_len:
                        chunked_raw_sents = self.heuristic_sentence_splitting(raw_sent)
                        if len(chunked_raw_sents) == 1:
                            continue
                        
                        for sent in chunked_raw_sents:
                            seg_sents.append(sent)
                    else:
                        seg_sents.append(raw_sent)
        else:
            raise NameError("*** Sentence splitter crashed, with trace %s..." % errdata)
            
        
        return seg_sents
     
    
    def sentence_splitter(self, raw_text):
        return sent_tokenize(raw_text)
    
    def preprocess_punctuation(self, text):
        '''
    
        '''
        '''
            replacing all numbers (float too) by zero
        '''
        text = re.sub(r'( )*([0-9])+\n\n+', '! ', text)
        text = re.sub(r'(([0-9]+)|([0-9]+(,[0-9]+)*\.[0-9]+))',"0",text)
        '''
            3 or more newline/whitespaces/punctuation = 2 newline/whitespaces/punctuation
            and, then, remove spaces and newlines between punctuation
        '''
        for chari in ["\n"," ","\?","!","\.",",",":",";"]:
            text = re.sub(r'(%s%s)(%s)+'%(chari,chari,chari),("%s%s"%(chari,chari)).replace('\\',''),text)
    
            text = re.sub(r'%s[ \n]+%s'%(chari,chari),"%s "%(chari).replace('\\',''),text)
    
        return text.replace("\n\n","\n").replace("  "," ")
    
                
if __name__ == '__main__':
    
    file_path = "/home/forrest/workspace/LINE/local_research/paraphrase_detection_lab/datasets/pan-plagiarism-corpus-2010/source-document/part7/source-document03067.txt"
    
#     file_path = "/home/forrest/workspace/LINE/lab/parser/RST-Style/gCRF_dist/src/ex1.txt"
    prep = Preprocessor()
    
    document = prep.raw_document_splitter(file_path)
    
    print(len(document))
