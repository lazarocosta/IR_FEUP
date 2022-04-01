import requests
import json
params = {"api-key":"0A51B92BLwrho7TvmF5b78H274U2FtcB"}
year=2019
month=1

x=2020
#while x = < 2022:
docs = []
receive = requests.get('https://api.nytimes.com/svc/archive/v1/'+ str(year) +'/'+ str(month) +'.json', params=params)
with open('records.json','a') as jsonFile:
    #jsonfile.write(receive.json())
    #f.write('ola')
    for doc in receive.json()['response']['docs']:
        newdoc= {
            "abstract": doc['abstract'],
            "snippet": doc['snippet'],
            "lead_paragraph": doc['lead_paragraph'],
            "keywords": doc["keywords"],
            "pub_date": doc['pub_date'],
            "word_count": doc['word_count'],
            "_id":doc['_id']
        }
        docs.append(newdoc)
    jsonFile.write(json.dumps(docs))

#print(receive.url)

#aDict = {"a":54, "b":87}
#jsonString = json.dumps(aDict)
#jsonFile = open("data.json", "a")
#jsonFile.write(jsonString)
#jsonFile.close()
