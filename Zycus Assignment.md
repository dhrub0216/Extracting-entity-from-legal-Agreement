Question- Extract various parameters from different types of rental agreement. 
Important points to notice to solve the problem- I have unlabelled dataset (Unstructured Data). 

Approach 1- Used Spacy NER function to solve the problem - Didn't work (Failed to recognize properly with substantial accuracy)

Approach 2- Used Spacy function creation to create a function for each parameters to be extracted- Didn't work out as there are different formats of housing agreements.

Approach 3- Most of models that are developed requires labelled data(which is not available in our case)- Rejected

Approach 4- We can apply only pre-trained models which would have been trained on similar type of data as ours as general NER(Named Entity Recognition) fails to recognize parameters in our case. We even cannot train pre-trained models on our custom data as those also need labels. 

Approach 5- Labelling data by yourself (Haven't tried it. Doubts about manual labelling vs pre-existing tools to help in labelling better just like LabelImg for images)


```
#First we are taking only few files so as to first build our approach easily
```


```
import pandas as pd
import glob #for reading multiple files from a folder
import docx 
import tabula  #for reading pdf file
import re #for regular expression
import docx2txt #for converting document into text
import nltk
import spacy
import en_core_web_sm
nlp = en_core_web_sm.load()

doc= docx2txt.process('E:/workspace-zycus/data/training/36199312-Rental-Agreement.pdf.docx')
doc_nlp= nlp(doc)
doc2= docx2txt.process('E:/workspace-zycus/data/training/288024755-Rental-Agreement-1.pdf.docx')
doc2_nlp= nlp(doc2)



```


```
spacy.displacy.serve(nlp(doc), style="ent")
```

    E:\Anaconda\lib\site-packages\spacy\displacy\__init__.py:94: UserWarning: [W011] It looks like you're calling displacy.serve from within a Jupyter notebook or a similar environment. This likely means you're already running a local web server, so there's no need to make displaCy start another one. Instead, you should be able to replace displacy.serve with displacy.render to show the visualization.
      warnings.warn(Warnings.W011)
    


<span class="tex2jax_ignore"><!DOCTYPE html>
<html lang="en">
    <head>
        <title>displaCy</title>
    </head>

    <body style="font-size: 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji', 'Segoe UI Symbol'; padding: 4rem 2rem; direction: ltr">
<figure style="margin-bottom: 6rem">
<div class="entities" style="line-height: 2.5; direction: ltr">RENEWAL OF RENTAL AGREEMENT</br></br>This AGREEMENT of Rent is made in 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Bangalore
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
 and 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Executed
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>

<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    today
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
 the lstth of 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    May 2010
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
</br></br>BY AND BETWEEN</br></br>1. Mr. 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Balaji
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
.R</br></br>Aged 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    about 63 years
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
,</br></br>No 24 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    2nd Cross
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
, SBM Colony Mathikere - 560054</br></br>Hereinafter referred and called as the ‘Lessor’ of the 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    First
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORDINAL</span>
</mark>
 part of 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    one
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
 part:</br></br>//AND//</br></br>1 Mr.Kartheek R Aged 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    about 31 years
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
,</br></br>No.81, sri manjunatha nilaya, raju colony, 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    yamalur


    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
B angalore-560037.</br></br>Hereinafter referred and called as the ‘
<mark class="entity" style="background: #f0d0ff; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Lessees
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">WORK_OF_ART</span>
</mark>
’ of the 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    second
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORDINAL</span>
</mark>
 part of the another part:</br></br></br>NOW THIS AGREEMENT OF RENT WITNESSETH TN AS FOLLOWS:</br></br>Whereas the 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    first
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORDINAL</span>
</mark>
 party is the sole and absolute owner of the above cited / scheduled premises is hereby continued to be rented out the same to the 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    second
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORDINAL</span>
</mark>
 party which terms and conditions is as follows:</br></br>The lessor have received a security Deposit amount of Rs.40,000/- (Rupees Fourty Thousand only) from the 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Lessees
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 and hereby acknowledges the receipt of the same, which carries no interest but to be returned to the lessee at the time of the lessee 
<mark class="entity" style="background: #ff9561; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Vacates
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">LOC</span>
</mark>
 and hands over the position.</br></br>The Rent is payable by the 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Lessees
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 to the Lessor is a sum of 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Rs.3800/- (Rupees Thirty
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>

<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Eight Thousand
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">TIME</span>
</mark>
 Only) on or before 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    10th
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORDINAL</span>
</mark>
 of every 
<mark class="entity" style="background: #ff8197; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    English
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">LANGUAGE</span>
</mark>
 Calendar Month.</br></br>This agreement is in force for a period of 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    eleven
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
 (
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    11) months
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
 and the same may be renewed by the mutual understanding of both the Lessor and the 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Lessee
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
.</br></br>In case of either party wants back the portion or vacates the portion either must be informed within 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    one month
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
 prior notice.,</br></br>The Electricity and water charged is to be borne by the 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Lessees
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
 only.</br></br>The 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Lessee
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 should neither sublet nor underlet and shall use the premises only for 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Residential
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 and for not any business purpose, should maintain clean tidy with no interruption, and shall handover at the time of vacating in tenantable condition.</br></br>SCHEDULE</br></br>The Schedule of Residential premises. 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    No.81
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
, sri manjunatha nilaya, raju colony, 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    yamalur Bangalore-560037
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
, consists of 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    one
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
 Hall, 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    One
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
 Bed Room, Kitchen, 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Bath Room
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
, and 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Completely Electrified
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 with running water facilities and other amenities.</br></br>In witness whereof both the parties have set their respective hands and affixed their signatures, hereunder the following and presence of the 
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    two
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
 witnesses 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    today
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>

<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    the day month year
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
 cited above.</br></br>WITNESSES:</br></br>	
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    1
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">CARDINAL</span>
</mark>
.	LESSOR/OWNER</br></br></br></br>2.LESSEES/
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    TENANT
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>
.</div>
</figure>
</body>
</html></span>


    
    Using the 'ent' visualizer
    Serving on http://0.0.0.0:5000 ...
    
    


```
from spacy import displacy

displacy.serve(doc2_nlp, style="ent")
```


```
#It is clear from running this on 3 to 4 documents that Approach 1 wouldn't work out
```


```
#APPROACH 2
```


```

#for finding the owner name (but spacy couldn't)
total_doc = nlp('This deed of rental agreement executed at Bangalore this fifth day of December 2008 between MR.K.Kuttan S/o Kelu Aehari (Late) residing at site No 152 Geethalayam OMH colong S.M. Road 1st main, T.Dasarahalli, Bangalore-57. Here after called the lessor which term shall where the contest so admits mean and includes his legal heirs representatives executors and assign and Sri. P.M. Narayana Namboodri “Laxmi Leela” ground floor 3rd cross Ayyappa Nagar behind Ayyappa Temple, Jalahalli West, Bangalore - 15 referred as tenant which term shall wherever the context so admits mean and includes his legal heirs representations executors assign witnesses.')

for tok in total_doc:
    print(tok.text, "...", tok.dep_,"...",tok.pos_)
```

    This ... det ... DET
    deed ... nsubj ... NOUN
    of ... prep ... ADP
    rental ... amod ... ADJ
    agreement ... pobj ... NOUN
    executed ... ROOT ... VERB
    at ... prep ... ADP
    Bangalore ... pobj ... PROPN
    this ... det ... DET
    fifth ... amod ... ADJ
    day ... npadvmod ... NOUN
    of ... prep ... ADP
    December ... pobj ... PROPN
    2008 ... nummod ... NUM
    between ... prep ... ADP
    MR.K.Kuttan ... punct ... PROPN
    S ... nmod ... PROPN
    / ... punct ... SYM
    o ... ROOT ... NOUN
    Kelu ... compound ... PROPN
    Aehari ... ROOT ... PROPN
    ( ... punct ... PUNCT
    Late ... appos ... ADV
    ) ... punct ... PUNCT
    residing ... ROOT ... VERB
    at ... prep ... ADP
    site ... pobj ... NOUN
    No ... ROOT ... PROPN
    152 ... pobj ... NUM
    Geethalayam ... nmod ... PROPN
    OMH ... compound ... PROPN
    colong ... compound ... ADJ
    S.M. ... compound ... PROPN
    Road ... compound ... PROPN
    1st ... ROOT ... NOUN
    main ... ROOT ... NOUN
    , ... punct ... PUNCT
    T.Dasarahalli ... npadvmod ... PROPN
    , ... punct ... PUNCT
    Bangalore-57 ... ROOT ... PROPN
    . ... punct ... PUNCT
    Here ... advmod ... ADV
    after ... advmod ... ADP
    called ... ROOT ... VERB
    the ... det ... DET
    lessor ... dobj ... NOUN
    which ... det ... DET
    term ... nsubj ... NOUN
    shall ... ccomp ... VERB
    where ... advmod ... ADV
    the ... det ... DET
    contest ... nsubj ... NOUN
    so ... advmod ... ADV
    admits ... aux ... VERB
    mean ... ccomp ... VERB
    and ... cc ... CCONJ
    includes ... conj ... VERB
    his ... poss ... DET
    legal ... amod ... ADJ
    heirs ... compound ... NOUN
    representatives ... compound ... NOUN
    executors ... dobj ... NOUN
    and ... cc ... CCONJ
    assign ... conj ... NOUN
    and ... cc ... CCONJ
    Sri ... conj ... PROPN
    . ... punct ... PUNCT
    P.M. ... nmod ... PROPN
    Narayana ... compound ... PROPN
    Namboodri ... ROOT ... PROPN
    “ ... punct ... PUNCT
    Laxmi ... nmod ... PROPN
    Leela ... nmod ... PROPN
    ” ... punct ... PUNCT
    ground ... compound ... NOUN
    floor ... compound ... NOUN
    3rd ... compound ... NOUN
    cross ... ROOT ... NOUN
    Ayyappa ... compound ... PROPN
    Nagar ... nsubj ... PROPN
    behind ... prep ... ADP
    Ayyappa ... compound ... PROPN
    Temple ... pobj ... PROPN
    , ... punct ... PUNCT
    Jalahalli ... compound ... PROPN
    West ... conj ... PROPN
    , ... punct ... PUNCT
    Bangalore ... compound ... PROPN
    - ... punct ... PUNCT
    15 ... conj ... NUM
    referred ... ROOT ... VERB
    as ... prep ... SCONJ
    tenant ... pobj ... NOUN
    which ... det ... DET
    term ... nsubj ... NOUN
    shall ... advcl ... VERB
    wherever ... advmod ... ADV
    the ... det ... DET
    context ... nsubj ... NOUN
    so ... advmod ... ADV
    admits ... aux ... VERB
    mean ... advcl ... VERB
    and ... cc ... CCONJ
    includes ... conj ... VERB
    his ... poss ... DET
    legal ... amod ... ADJ
    heirs ... compound ... NOUN
    representations ... compound ... NOUN
    executors ... compound ... NOUN
    assign ... compound ... VERB
    witnesses ... dobj ... NOUN
    . ... punct ... PUNCT
    


```
print([(ent.text, ent.label_) for ent in total_doc.ents])
```

    [('Bangalore', 'GPE'), ('this fifth day of December 2008', 'DATE'), ('152', 'CARDINAL'), ('Geethalayam', 'NORP'), ('OMH', 'ORG'), ('T.Dasarahalli', 'WORK_OF_ART'), ('P.M. Narayana Namboodri', 'PERSON'), ('Laxmi Leela', 'WORK_OF_ART'), ('3rd', 'CARDINAL'), ('Ayyappa Nagar', 'PERSON'), ('Ayyappa Temple', 'PERSON'), ('Jalahalli West', 'GPE')]
    


```
#Finding Data parameter
for ent in doc2_nlp.ents:
    if(ent.label_=="DATE"):
        print(ent.text, ent.start_char, ent.end_char, ent.label_) 
```


```
#Data 2
total_doc2= nlp('THIS AGREEMENT OF RENTAL LEASE made and entered into on this the 15th day of February 2012 (15.02.2012).By and betweenMr M.SANTOSH S/O.M.DATTATREYA SASTRY, aged 31 years, having permanent residence at Flat	FI,Goldenpearl,No:3,V.O.C.Street,Near SAN Academy School,Padmavathy Nagar,Velachery,Chennai-600042., hereinafter referred to as the “LANDLORD” (Which term shall mean and include her legal heirs, executors, administrators, legal representatives and assigns) of ONE PART.ANDMr. Jeyanth B, S/o. Balasundaram C aged 30 years, working as a software Engineer, Cognizant Technology Solutions ,Chennai., having permanent residence at No 5/856, Lakeview 3rd Street, Iyyappa Nagar, Madipakkam Chennai - 91 hereinafter referred to as the TENANT”(which term shall mean and include his legal heirs, successors, executors, administrators, legal representatives and assigns) of THE OTHER PART.')

for tok in total_doc2:
    print(tok.text, "...", tok.dep_,"...", tok.pos_)
```

    THIS ... det ... DET
    AGREEMENT ... nsubj ... PROPN
    OF ... prep ... ADP
    RENTAL ... compound ... PROPN
    LEASE ... pobj ... PROPN
    made ... ROOT ... VERB
    and ... cc ... CCONJ
    entered ... conj ... VERB
    into ... prep ... ADP
    on ... prep ... ADP
    this ... pobj ... DET
    the ... det ... DET
    15th ... amod ... ADJ
    day ... npadvmod ... NOUN
    of ... prep ... ADP
    February ... pobj ... PROPN
    2012 ... nummod ... NUM
    ( ... punct ... PUNCT
    15.02.2012).By ... ROOT ... NUM
    and ... cc ... CCONJ
    betweenMr ... conj ... PROPN
    M.SANTOSH ... compound ... PROPN
    S ... nmod ... PROPN
    / ... punct ... SYM
    O.M.DATTATREYA ... appos ... PROPN
    SASTRY ... ROOT ... PROPN
    , ... punct ... PUNCT
    aged ... amod ... VERB
    31 ... nummod ... NUM
    years ... npadvmod ... NOUN
    , ... punct ... PUNCT
    having ... acl ... VERB
    permanent ... amod ... ADJ
    residence ... dobj ... NOUN
    at ... prep ... ADP
    Flat ... amod ... PROPN
    	 ...  ... SPACE
    FI ... pobj ... PROPN
    , ... punct ... PUNCT
    Goldenpearl ... npadvmod ... PROPN
    , ... punct ... PUNCT
    No:3,V.O.C.Street ... appos ... NOUN
    , ... punct ... PUNCT
    Near ... compound ... PROPN
    SAN ... compound ... PROPN
    Academy ... compound ... PROPN
    School ... ROOT ... PROPN
    , ... punct ... PUNCT
    Padmavathy ... compound ... PROPN
    Nagar ... conj ... PROPN
    , ... punct ... PUNCT
    Velachery ... conj ... PROPN
    , ... punct ... PUNCT
    Chennai-600042 ... dep ... PROPN
    . ... ROOT ... PROPN
    , ... punct ... PUNCT
    hereinafter ... nsubj ... PROPN
    referred ... ROOT ... VERB
    to ... prep ... ADP
    as ... prep ... SCONJ
    the ... det ... DET
    “ ... punct ... PUNCT
    LANDLORD ... pobj ... PROPN
    ” ... punct ... PUNCT
    ( ... punct ... PUNCT
    Which ... det ... DET
    term ... nsubj ... NOUN
    shall ... aux ... VERB
    mean ... relcl ... VERB
    and ... cc ... CCONJ
    include ... conj ... VERB
    her ... poss ... DET
    legal ... amod ... ADJ
    heirs ... dobj ... NOUN
    , ... punct ... PUNCT
    executors ... conj ... NOUN
    , ... punct ... PUNCT
    administrators ... conj ... NOUN
    , ... punct ... PUNCT
    legal ... amod ... ADJ
    representatives ... conj ... NOUN
    and ... cc ... CCONJ
    assigns ... conj ... NOUN
    ) ... punct ... PUNCT
    of ... prep ... ADP
    ONE ... compound ... PROPN
    PART.ANDMr ... pobj ... NOUN
    . ... punct ... PUNCT
    Jeyanth ... compound ... PROPN
    B ... dep ... PROPN
    , ... punct ... PUNCT
    S ... nmod ... PROPN
    / ... punct ... SYM
    o ... ROOT ... NOUN
    . ... punct ... PUNCT
    Balasundaram ... compound ... PROPN
    C ... nsubj ... PROPN
    aged ... ROOT ... VERB
    30 ... nummod ... NUM
    years ... npadvmod ... NOUN
    , ... punct ... PUNCT
    working ... advcl ... VERB
    as ... prep ... SCONJ
    a ... det ... DET
    software ... compound ... NOUN
    Engineer ... pobj ... PROPN
    , ... punct ... PUNCT
    Cognizant ... compound ... PROPN
    Technology ... compound ... PROPN
    Solutions ... conj ... PROPN
    , ... punct ... PUNCT
    Chennai ... nsubj ... PROPN
    . ... punct ... PUNCT
    , ... punct ... PUNCT
    having ... advcl ... VERB
    permanent ... amod ... ADJ
    residence ... dobj ... NOUN
    at ... prep ... ADP
    No ... det ... NOUN
    5/856 ... pobj ... PROPN
    , ... punct ... PUNCT
    Lakeview ... compound ... PROPN
    3rd ... compound ... PROPN
    Street ... conj ... PROPN
    , ... punct ... PUNCT
    Iyyappa ... compound ... PROPN
    Nagar ... conj ... PROPN
    , ... punct ... PUNCT
    Madipakkam ... compound ... PROPN
    Chennai ... compound ... PROPN
    - ... punct ... PUNCT
    91 ... nummod ... NUM
    hereinafter ... conj ... NOUN
    referred ... ROOT ... VERB
    to ... prep ... ADP
    as ... mark ... SCONJ
    the ... det ... DET
    TENANT”(which ... compound ... PROPN
    term ... nsubj ... NOUN
    shall ... aux ... VERB
    mean ... advcl ... VERB
    and ... cc ... CCONJ
    include ... conj ... VERB
    his ... poss ... DET
    legal ... amod ... ADJ
    heirs ... dobj ... NOUN
    , ... punct ... PUNCT
    successors ... conj ... NOUN
    , ... punct ... PUNCT
    executors ... conj ... NOUN
    , ... punct ... PUNCT
    administrators ... conj ... NOUN
    , ... punct ... PUNCT
    legal ... amod ... ADJ
    representatives ... conj ... NOUN
    and ... cc ... CCONJ
    assigns ... conj ... NOUN
    ) ... punct ... PUNCT
    of ... prep ... ADP
    THE ... det ... DET
    OTHER ... amod ... PROPN
    PART ... pobj ... PROPN
    . ... punct ... PUNCT
    


```
#Writing function to select particular parameter (Failed as each document format were different)
ent1=""
ent2=""
prv_tok_dep=""
prv_tok_text=""
prefix=""
modifier=""
    
for tok in total_doc2:
    
    if tok.dep_ in ["compound","punct"] :
        if prv_tok_dep in ["conj","prep"] :
            modifier = prv_tok_text
    if tok.dep_=="punct":
        if prv_tok_dep=="nmod":
            print(modifier)
    
            

    prv_tok_dep = tok.dep_
    prv_tok_text = tok.text


```

    betweenMr
    of
    


```

```


```

from spacy.matcher import Matcher 
from spacy.tokens import Span 

pattern = [{'POS':'conj'}, 
           {'LOWER': 'such'}, 
           {'LOWER': 'as'}, 
           {'POS': 'nmod'},
           #proper noun]
           
           
# Matcher class object 
matcher = Matcher(nlp.vocab) 
matcher.add("matching_1", None, pattern) 

matches = matcher(total_doc2)
span = total_doc2[matches[0][1]:matches[0][2]] 

print(span.text)
```


```
#Approach 4- Using pre-trained deeppavlov model
from deeppavlov import configs, build_model
deeppavlov_ner = build_model(configs.ner.ner_ontonotes, download=True)
deeppavlov_ner([doc])
```

    2020-07-24 01:23:58.741 INFO in 'deeppavlov.download'['download'] at line 132: Skipped http://files.deeppavlov.ai/deeppavlov_data/ner_ontonotes_v3_cpu_compatible.tar.gz download because of matching hashes
    2020-07-24 01:24:00.453 INFO in 'deeppavlov.download'['download'] at line 132: Skipped http://files.deeppavlov.ai/embeddings/glove.6B.100d.txt download because of matching hashes
    2020-07-24 01:24:00.855 INFO in 'deeppavlov.core.data.simple_vocab'['simple_vocab'] at line 115: [loading vocabulary from C:\Users\RAM G\.deeppavlov\models\ner_ontonotes\tag.dict]
    2020-07-24 01:24:00.859 INFO in 'deeppavlov.core.data.simple_vocab'['simple_vocab'] at line 115: [loading vocabulary from C:\Users\RAM G\.deeppavlov\models\ner_ontonotes\char.dict]
    2020-07-24 01:24:00.862 INFO in 'deeppavlov.models.embedders.glove_embedder'['glove_embedder'] at line 52: [loading GloVe embeddings from `C:\Users\RAM G\.deeppavlov\downloads\embeddings\glove.6B.100d.txt`]
    2020-07-24 01:25:13.533 INFO in 'deeppavlov.core.layers.tf_layers'['tf_layers'] at line 760: 
    Warning! tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell is used. It is okay for inference mode, but if you train your model with this cell it could NOT be used with tf.contrib.cudnn_rnn.CudnnLSTMCell later. 
    2020-07-24 01:25:13.648 INFO in 'deeppavlov.core.layers.tf_layers'['tf_layers'] at line 760: 
    Warning! tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell is used. It is okay for inference mode, but if you train your model with this cell it could NOT be used with tf.contrib.cudnn_rnn.CudnnLSTMCell later. 
    2020-07-24 01:25:15.948 INFO in 'deeppavlov.core.models.tf_model'['tf_model'] at line 51: [loading model from C:\Users\RAM G\.deeppavlov\models\ner_ontonotes\model]
    

    INFO:tensorflow:Restoring parameters from C:\Users\RAM G\.deeppavlov\models\ner_ontonotes\model
    




    [[['RENEWAL',
       'OF',
       'RENTAL',
       'AGREEMENT',
       'This',
       'AGREEMENT',
       'of',
       'Rent',
       'is',
       'made',
       'in',
       'Bangalore',
       'and',
       'Executed',
       'today',
       'the',
       'lstth',
       'of',
       'May',
       '2010',
       'BY',
       'AND',
       'BETWEEN',
       '1',
       '.',
       'Mr.',
       'Balaji.R',
       'Aged',
       'about',
       '63',
       'years',
       ',',
       'No',
       '24',
       '2nd',
       'Cross',
       ',',
       'SBM',
       'Colony',
       'Mathikere',
       '-',
       '560054',
       'Hereinafter',
       'referred',
       'and',
       'called',
       'as',
       'the',
       '‘',
       'Lessor',
       '’',
       'of',
       'the',
       'First',
       'part',
       'of',
       'one',
       'part',
       ':',
       '//AND//',
       '1',
       'Mr.Kartheek',
       'R',
       'Aged',
       'about',
       '31',
       'years',
       ',',
       'No.81',
       ',',
       'sri',
       'manjunatha',
       'nilaya',
       ',',
       'raju',
       'colony',
       ',',
       'yamalur',
       'B',
       'angalore-560037',
       '.',
       'Hereinafter',
       'referred',
       'and',
       'called',
       'as',
       'the',
       '‘',
       'Lessees',
       '’',
       'of',
       'the',
       'second',
       'part',
       'of',
       'the',
       'another',
       'part',
       ':',
       'NOW',
       'THIS',
       'AGREEMENT',
       'OF',
       'RENT',
       'WITNESSETH',
       'TN',
       'AS',
       'FOLLOWS',
       ':',
       'Whereas',
       'the',
       'first',
       'party',
       'is',
       'the',
       'sole',
       'and',
       'absolute',
       'owner',
       'of',
       'the',
       'above',
       'cited',
       '/',
       'scheduled',
       'premises',
       'is',
       'hereby',
       'continued',
       'to',
       'be',
       'rented',
       'out',
       'the',
       'same',
       'to',
       'the',
       'second',
       'party',
       'which',
       'terms',
       'and',
       'conditions',
       'is',
       'as',
       'follows',
       ':',
       'The',
       'lessor',
       'have',
       'received',
       'a',
       'security',
       'Deposit',
       'amount',
       'of',
       'Rs.40,000/-',
       '(',
       'Rupees',
       'Fourty',
       'Thousand',
       'only',
       ')',
       'from',
       'the',
       'Lessees',
       'and',
       'hereby',
       'acknowledges',
       'the',
       'receipt',
       'of',
       'the',
       'same',
       ',',
       'which',
       'carries',
       'no',
       'interest',
       'but',
       'to',
       'be',
       'returned',
       'to',
       'the',
       'lessee',
       'at',
       'the',
       'time',
       'of',
       'the',
       'lessee',
       'Vacates',
       'and',
       'hands',
       'over',
       'the',
       'position',
       '.',
       'The',
       'Rent',
       'is',
       'payable',
       'by',
       'the',
       'Lessees',
       'to',
       'the',
       'Lessor',
       'is',
       'a',
       'sum',
       'of',
       'Rs.3800/-',
       '(',
       'Rupees',
       'Thirty',
       'Eight',
       'Thousand',
       'Only',
       ')',
       'on',
       'or',
       'before',
       '10th',
       'of',
       'every',
       'English',
       'Calendar',
       'Month',
       '.',
       'This',
       'agreement',
       'is',
       'in',
       'force',
       'for',
       'a',
       'period',
       'of',
       'eleven',
       '(',
       '11',
       ')',
       'months',
       'and',
       'the',
       'same',
       'may',
       'be',
       'renewed',
       'by',
       'the',
       'mutual',
       'understanding',
       'of',
       'both',
       'the',
       'Lessor',
       'and',
       'the',
       'Lessee',
       '.',
       'In',
       'case',
       'of',
       'either',
       'party',
       'wants',
       'back',
       'the',
       'portion',
       'or',
       'vacates',
       'the',
       'portion',
       'either',
       'must',
       'be',
       'informed',
       'within',
       'one',
       'month',
       'prior',
       'notice.',
       ',',
       'The',
       'Electricity',
       'and',
       'water',
       'charged',
       'is',
       'to',
       'be',
       'borne',
       'by',
       'the',
       'Lessees',
       'only',
       '.',
       'The',
       'Lessee',
       'should',
       'neither',
       'sublet',
       'nor',
       'underlet',
       'and',
       'shall',
       'use',
       'the',
       'premises',
       'only',
       'for',
       'Residential',
       'and',
       'for',
       'not',
       'any',
       'business',
       'purpose',
       ',',
       'should',
       'maintain',
       'clean',
       'tidy',
       'with',
       'no',
       'interruption',
       ',',
       'and',
       'shall',
       'handover',
       'at',
       'the',
       'time',
       'of',
       'vacating',
       'in',
       'tenantable',
       'condition',
       '.',
       'SCHEDULE',
       'The',
       'Schedule',
       'of',
       'Residential',
       'premises',
       '.',
       'No.81',
       ',',
       'sri',
       'manjunatha',
       'nilaya',
       ',',
       'raju',
       'colony',
       ',',
       'yamalur',
       'Bangalore-560037',
       ',',
       'consists',
       'of',
       'one',
       'Hall',
       ',',
       'One',
       'Bed',
       'Room',
       ',',
       'Kitchen',
       ',',
       'Bath',
       'Room',
       ',',
       'and',
       'Completely',
       'Electrified',
       'with',
       'running',
       'water',
       'facilities',
       'and',
       'other',
       'amenities',
       '.',
       'In',
       'witness',
       'whereof',
       'both',
       'the',
       'parties',
       'have',
       'set',
       'their',
       'respective',
       'hands',
       'and',
       'affixed',
       'their',
       'signatures',
       ',',
       'hereunder',
       'the',
       'following',
       'and',
       'presence',
       'of',
       'the',
       'two',
       'witnesses',
       'today',
       'the',
       'day',
       'month',
       'year',
       'cited',
       'above',
       '.',
       'WITNESSES',
       ':',
       '1',
       '.',
       'LESSOR/OWNER',
       '2.LESSEES/TENANT',
       '.']],
     [['O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'B-GPE',
       'O',
       'O',
       'B-DATE',
       'O',
       'O',
       'O',
       'B-DATE',
       'I-DATE',
       'O',
       'O',
       'O',
       'B-CARDINAL',
       'O',
       'O',
       'B-PERSON',
       'O',
       'B-DATE',
       'I-DATE',
       'I-DATE',
       'O',
       'B-CARDINAL',
       'I-CARDINAL',
       'I-CARDINAL',
       'O',
       'O',
       'B-LAW',
       'I-LAW',
       'I-LAW',
       'I-LAW',
       'I-LAW',
       'I-LAW',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'B-ORG',
       'I-ORG',
       'O',
       'O',
       'B-ORDINAL',
       'O',
       'O',
       'B-CARDINAL',
       'O',
       'O',
       'O',
       'B-DATE',
       'I-DATE',
       'I-DATE',
       'I-DATE',
       'I-DATE',
       'I-DATE',
       'I-DATE',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'B-ORG',
       'I-ORG',
       'O',
       'O',
       'B-ORDINAL',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'B-ORDINAL',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'B-ORDINAL',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'B-PRODUCT',
       'I-PRODUCT',
       'O',
       'O',
       'O',
       'O',
       'B-ORG',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'B-PERSON',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'B-ORG',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'B-QUANTITY',
       'I-QUANTITY',
       'I-QUANTITY',
       'I-QUANTITY',
       'O',
       'O',
       'O',
       'O',
       'B-ORDINAL',
       'O',
       'O',
       'B-LANGUAGE',
       'O',
       'B-DATE',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'B-DATE',
       'I-DATE',
       'I-DATE',
       'I-DATE',
       'I-DATE',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'B-ORG',
       'O',
       'O',
       'B-ORG',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'B-DATE',
       'I-DATE',
       'O',
       'O',
       'O',
       'B-ORG',
       'I-ORG',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'B-ORG',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'B-EVENT',
       'I-EVENT',
       'I-EVENT',
       'I-EVENT',
       'I-EVENT',
       'I-EVENT',
       'I-EVENT',
       'I-EVENT',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'B-CARDINAL',
       'O',
       'O',
       'O',
       'B-CARDINAL',
       'B-ORG',
       'O',
       'B-ORG',
       'I-ORG',
       'I-ORG',
       'I-ORG',
       'I-ORG',
       'I-ORG',
       'I-ORG',
       'I-ORG',
       'O',
       'O',
       'B-FAC',
       'I-FAC',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'O',
       'B-CARDINAL',
       'O',
       'O',
       'B-DATE',
       'I-DATE',
       'I-DATE',
       'I-DATE',
       'O',
       'O',
       'O',
       'O',
       'O',
       'B-CARDINAL',
       'O',
       'O',
       'O',
       'O']]]




```
#Approach 4- Using Polyglot model
from polyglot.text import Text
Text(doc_nlp).entities
```


```
import docx2txt
doc= docx2txt.process('36199312-Rental-Agreement.pdf.docx')
```


```
from allennlp.predictors import Predictor
al = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/ner-model-2020.02.10.tar.gz")
```


```
al.predict(sentence=doc)
```


```
#Now we need labelled dataset to use lstm to train better.
```
