from urllib.request import urlopen
from lxml import etree

PMID = '???' #nr pracy
baseurl = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?"
query = "db=pubmed&id="+PMID+"&format=xml"    
url = baseurl+query

f = urlopen(url) #otwieram połączenie URL
resultxml = f.read() #czytam zawartość
xml = etree.XML(resultxml) #strukturyzacja

resultelements= xml.xpath("//ArticleTitle")   #Inne: ArticleTitle, LastName, Keyword, AbstractText, DescriptorName

for element in resultelements:
    print(element.text)