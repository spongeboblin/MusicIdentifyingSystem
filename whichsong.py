import json
from recognizer import SimpleRecognizer
from getfeature import SFA_HPCP_Extractor

if __name__ == '__main__':
	while (1) :
		print "Type 1 to start searching or 0 to quit >>>"
		op = raw_input()
		if op == '0' :
			break
		print "Type the name of the file you want to search >>>"
		fileName = raw_input()
		
		extractor = SFA_HPCP_Extractor()

		with open('result.json', 'r')  as f :
			rawData = json.load(f)

		reg = SimpleRecognizer(rawData)

		#print fileName

		result = extractor.ExtractFromFile(filename="./"+fileName,ratio=0.7)
		R = []
		for E in result :
			feature = E[1]
			newTuple = [feature, E[0]]
			R.append(newTuple)
		#print R
       #print data.database
		print reg.recognize(R)

