"""TF-IDF retrieval. 2026-06-01"""
from typing import List, Tuple
import math
from collections import Counter

class TFIDFBackend:
    def __init__(self): self.corpus=[]; self.idf={}; self._docs=[]
    def fit(self,docs:List[str]):
        self._docs=docs; self.corpus=[d.lower().split() for d in docs]
        N=len(docs); df=Counter(t for doc in self.corpus for t in set(doc))
        self.idf={t:math.log((N+1)/(f+1))+1 for t,f in df.items()}
    def retrieve(self,query:str,top_k=5)->List[Tuple[int,float]]:
        q=query.lower().split(); scores=[]
        for i,doc in enumerate(self.corpus):
            freq=Counter(doc); dl=len(doc)+1
            s=sum(self.idf.get(t,0)*(freq[t]/dl) for t in q)
            scores.append((i,s))
        return sorted(scores,key=lambda x:x[1],reverse=True)[:top_k]

# ts:2026-06-01T11:45:00
