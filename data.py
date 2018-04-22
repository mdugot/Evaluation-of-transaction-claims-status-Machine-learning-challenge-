import numpy as np
from tqdm import tqdm
import random

types = [
	"int32", #ID
	"U30", #SHIPPING_MODE
	"U30", #SHIPPING_PRICE
	"U30", #WARRANTIES_FLG
	"U30", #WARRANTIES_PRICE
	"float32", #CARD_PAYMENT
	"float32", #COUPON_PAYMENT
	"float32", #RSP_PAYMENT
	"float32", #WALLET_PAYMENT
	"U30", #PRICECLUB_STATUS
	"float32", #REGISTRATION_DATE
	"U30", #PURCHASE_COUNT
	"float32", #BUYER_BIRTHDAY_DATE
	"int32", #BUYER_DEPARTMENT
	"U30", #BUYING_DATE
	"U30", #SELLER_SCORE_COUNT
	"float32", #SELLER_SCORE_AVERAGE
	"U30", #SELLER_COUNTRY
	"int32", #SELLER_DEPARTMENT
	"U30", #PRODUCT_TYPE
	"U30", #PRODUCT_FAMILY
	"U30" #ITEM_PRICE
]

shippingMode = {
	'': 						[0,0,0,0,0,0,0,0,0,0,0,0],
	'CHRONOPOST': 				[1,0,0,0,0,0,0,0,0,0,0,1],
	'EXPRESS_DELIVERY': 		[0,1,0,0,0,0,0,0,0,0,0,1],
	'Kiala': 					[0,0,1,0,0,0,0,0,0,0,0,1],
	'MONDIAL_RELAY': 			[0,0,0,1,0,0,0,0,0,0,0,1],
	'MONDIAL_RELAY_PREPAYE': 	[0,0,0,0,1,0,0,0,0,0,0,1],
	'NORMAL': 					[0,0,0,0,0,1,0,0,0,0,0,1],
	'PICKUP': 					[0,0,0,0,0,0,1,0,0,0,0,1],
	'RECOMMANDE': 				[0,0,0,0,0,0,0,1,0,0,0,1],
	'SO_POINT_RELAIS': 			[0,0,0,0,0,0,0,0,1,0,0,1],
	'SO_RECOMMANDE': 			[0,0,0,0,0,0,0,0,0,1,0,1],
	'SUIVI':			 		[0,0,0,0,0,0,0,0,0,0,1,1]
}

shippingPrice = {
	'': 		[0,0,0,0,0,0],#[0, 0],
	'<1': 		[1,1,0,0,0,0],#[1, 0.5],
	'1<5': 		[1,0,1,0,0,0],#[1, 2.5],
	'5<10': 	[1,0,0,1,0,0],#[1, 7.5],
	'10<20':	[1,0,0,0,1,0],#[1, 15.0],
	'>20': 		[1,0,0,0,0,1]#[1, 25.0]
}

warrantiesFlg = {
	'False': 		[0],
	'True': 		[1]
}

warrantiesPrice = {
	'': 		[0,0,0,0,0,0],#[0, 0],
	'<5': 		[1,1,0,0,0,0],#[1, 2.5],
	'5<20': 	[1,0,1,0,0,0],#[1, 12.5],
	'20<50':	[1,0,0,1,0,0],#[1, 35],
	'50<100':	[1,0,0,0,1,0],#[1, 75],
	'100<500':	[1,0,0,0,0,1]#[1, 300]
}

priceclubStatus = {
	'': 			[0,0,0,0,0,0],
	'GOLD':			[1,1,0,0,0,0],
	'PLATINUM':		[1,0,1,0,0,0],
	'REGULAR':		[1,0,0,1,0,0],
	'SILVER':		[1,0,0,0,1,0],
	'UNSUBSCRIBED':	[1,0,0,0,0,1]
}

purchaseCount = {
	'<5': 		[1,0,0,0,0,0],#[2.5],
	'5<20': 	[0,1,0,0,0,0],#[12.5],
	'20<50':	[0,0,1,0,0,0],#[35],
	'50<100':	[0,0,0,1,0,0],#[75],
	'100<500':	[0,0,0,0,1,0],#[300],
	'>500':		[0,0,0,0,0,1]#[750]
}

buyingDate = {
	'1/2017': 		[1,0,0,0,0,0,0,0,0,0],#[1],
	'2/2017': 		[0,1,0,0,0,0,0,0,0,0],#[2],
	'3/2017': 		[0,0,1,0,0,0,0,0,0,0],#[3],
	'4/2017': 		[0,0,0,1,0,0,0,0,0,0],#[4],
	'5/2017': 		[0,0,0,0,1,0,0,0,0,0],#[5],
	'6/2017': 		[0,0,0,0,0,1,0,0,0,0],#[6],
	'7/2017': 		[0,0,0,0,0,0,1,0,0,0],#[7],
	'8/2017': 		[0,0,0,0,0,0,0,1,0,0],#[8],
	'9/2017': 		[0,0,0,0,0,0,0,0,1,0],#[9],
	'10/2017': 		[0,0,0,0,0,0,0,0,0,1]#[10]
}

sellerScoreCount = {
	'': 				[0, 0,0,0,0,0],
	'<100': 			[1, 1,0,0,0,0],
	'100<1000':			[1, 0,1,0,0,0],
	'1000<10000':		[1, 0,0,1,0,0],
	'10000<100000':		[1, 0,0,0,1,0],
	'100000<1000000':	[1, 0,0,0,0,1]
}

itemPrice = {
	'<10': 			[1,0,0,0,0,0,0,0],#[5],
	'10<20': 		[0,1,0,0,0,0,0,0],#[15],
	'20<50': 		[0,0,1,0,0,0,0,0],#[35],
	'50<100': 		[0,0,0,1,0,0,0,0],#[75],
	'100<500': 		[0,0,0,0,1,0,0,0],#[300],
	'500<1000':		[0,0,0,0,0,1,0,0],#[750],
	'1000<5000':	[0,0,0,0,0,0,1,0],#[3000],
	'>5000':		[0,0,0,0,0,0,0,1]#[10000]
}

claimType = {
	'UNDEFINED': 					[1,0,0,0,0,0,0,0],
	'DAMAGED': 						[0,1,0,0,0,0,0,0],
	'DIFFERENT': 					[0,0,1,0,0,0,0,0],
	'FAKE': 						[0,0,0,1,0,0,0,0],
	'NOT_RECEIVED': 				[0,0,0,0,1,0,0,0],
	'SELLER_CANCEL_POSTERIORI': 	[0,0,0,0,0,1,0,0],
	'WITHDRAWAL': 					[0,0,0,0,0,0,1,0],
	'-': 							[0,0,0,0,0,0,0,1]
}

class Data:
	
	def __init__(self):
		print("load training data")
		self.dataframe_train = np.genfromtxt("input_train.csv", delimiter=",", names = True, dtype=types)
		print("load testing data")
		self.dataframe_test = np.genfromtxt("input_test.csv", delimiter=",", names = True, dtype=types)
		print("load labels data")
		self.dataframe_labels = np.genfromtxt("output.csv", delimiter=";", names = True, dtype=["int32", "U30"])

		self.dataframe_all = np.append(self.dataframe_train, self.dataframe_test)

		self.shippingMode = shippingMode
		self.shippingPrice = shippingPrice
		self.warrantiesFlg = warrantiesFlg
		self.warrantiesPrice = warrantiesPrice
		self.priceclubStatus = priceclubStatus
		self.purchaseCount = purchaseCount
		self.buyingDate = buyingDate
		self.sellerScoreCount = sellerScoreCount
		self.itemPrice = itemPrice
		self.country = self.createData(np.unique(self.dataframe_all["SELLER_COUNTRY"]))
		self.productType = self.createData(np.unique(self.dataframe_all["PRODUCT_TYPE"]))
		self.productFamily = self.createData(np.unique(self.dataframe_all["PRODUCT_FAMILY"]))

		self.inputlen = len(self.structToData(self.dataframe_train[0], 0, 0, 0))

		self.claimType = claimType
		self.outputlen = len(self.claimType)

		print("prepare training data")
		self.allTraining = self.prepareAllData(self.dataframe_train)
		self.training = self.allTraining[:-5000]
		self.validation = self.allTraining[-5000:]
		print("prepare testing data")
		self.testing = self.prepareAllData(self.dataframe_test)
		print("prepare labels data")
		self.allLabels = self.prepareLabels()
		self.trainingLabels = self.allLabels[:-5000]
		self.validationLabels = self.allLabels[-5000:]

		self.dataByClaim()
		self.probs = dict()
		for key in self.claimType:
			self.probs[key] = len(self.dataframe_labels[self.dataframe_labels["CLAIM_TYPE"] == key]) / len(self.dataframe_labels)

	def createData(self, uniqueValues):
		data = dict()
		for i in range(len(uniqueValues)):
			data[uniqueValues[i]] = np.zeros(len(uniqueValues))
			data[uniqueValues[i]][i] = 1
		return data
	
	def nanvalue(self, value, nvalue):
		known = 1
		if np.isnan(value):
			known = 0
		return [known, nvalue]
	
	def normalize(self, allValues, minvalue, maxvalue):
		allValues[np.isnan(allValues)] = np.nanmean(allValues)
		gap = maxvalue - minvalue
		nvalues = (allValues - minvalue) / gap
		return nvalues
	
	def departement(self, value):
		data = np.zeros([101])
		data[value] = 1
		return data

	def structToData(self, struct, nregistration, nbirthday, nscore):
		data = np.array([], dtype="float32")
		data = np.append(data, self.shippingMode[struct["SHIPPING_MODE"]])
		data = np.append(data, self.shippingPrice[struct["SHIPPING_PRICE"]])
		data = np.append(data, self.warrantiesFlg[struct["WARRANTIES_FLG"]])
		data = np.append(data, self.warrantiesPrice[struct["WARRANTIES_PRICE"]])
		data = np.append(data, [struct["CARD_PAYMENT"]])
		data = np.append(data, [struct["COUPON_PAYMENT"]])
		data = np.append(data, [struct["RSP_PAYMENT"]])
		data = np.append(data, [struct["WALLET_PAYMENT"]])
		data = np.append(data, self.priceclubStatus[struct["PRICECLUB_STATUS"]])
		data = np.append(data, self.nanvalue(struct["REGISTRATION_DATE"], nregistration))
		data = np.append(data, self.purchaseCount[struct["PURCHASE_COUNT"]])
		data = np.append(data, self.nanvalue(struct["BUYER_BIRTHDAY_DATE"], nbirthday))
		data = np.append(data, self.departement(struct["BUYER_DEPARTMENT"]))
		data = np.append(data, self.buyingDate[struct["BUYING_DATE"]])
		data = np.append(data, self.sellerScoreCount[struct["SELLER_SCORE_COUNT"]])
		data = np.append(data, self.nanvalue(struct["SELLER_SCORE_AVERAGE"], nscore))
		data = np.append(data, self.country[struct["SELLER_COUNTRY"]])
		data = np.append(data, self.departement(struct["SELLER_DEPARTMENT"]))
		data = np.append(data, self.productType[struct["PRODUCT_TYPE"]])
		data = np.append(data, self.productFamily[struct["PRODUCT_FAMILY"]])
		data = np.append(data, self.itemPrice[struct["ITEM_PRICE"]])

		return data
	
	def prepareLabels(self):
		data = np.zeros([len(self.dataframe_labels), self.outputlen])
		for i in tqdm(range(len(self.dataframe_labels))):
			data[i] = self.claimType[self.dataframe_labels[i]["CLAIM_TYPE"]]
		return data
	
	def prepareAllData(self, dataframe):

		nregistration = self.normalize(dataframe["REGISTRATION_DATE"],
			np.nanmin(self.dataframe_train["REGISTRATION_DATE"]),
			np.nanmax(self.dataframe_train["REGISTRATION_DATE"]))
		nbirthday = self.normalize(dataframe["BUYER_BIRTHDAY_DATE"],
			np.nanmin(self.dataframe_train["BUYER_BIRTHDAY_DATE"]),
			np.nanmax(self.dataframe_train["BUYER_BIRTHDAY_DATE"]))
		nscore = self.normalize(dataframe["SELLER_SCORE_AVERAGE"],
			np.nanmin(self.dataframe_train["SELLER_SCORE_AVERAGE"]),
			np.nanmax(self.dataframe_train["SELLER_SCORE_AVERAGE"]))

		data = np.zeros([len(dataframe), self.inputlen])
		for i in tqdm(range(len(dataframe))):
			data[i] = self.structToData(dataframe[i], nregistration[i], nbirthday[i], nscore[i])
		return data

	def addNegative(self, claim, data):
		for key in self.claimType:
			if claim != key:
				self.negativeTraining[key].append(data)
				
	
	def dataByClaim(self):
		self.positiveTraining = dict()
		self.negativeTraining = dict()
		for key in self.claimType:
			self.positiveTraining[key] = []
			self.negativeTraining[key] = []
		for i in tqdm(range(len(self.training))):
			claimType = self.dataframe_labels["CLAIM_TYPE"][i]
			self.positiveTraining[claimType].append(self.training[i])
			self.addNegative(claimType, self.training[i])
			
	
	def shuffleTraining(self):
		combined = list(zip(self.training, self.trainingLabels))
		random.shuffle(combined)
		self.training, self.trainingLabels = zip(*combined)

	def getClaim(self, index):
		for key in self.claimType:	
			if (self.claimType[key][index] == 1):
				return key
		print("Error : Claim not found !")

	def balancedBatch(self, size):
		
		inputs = np.zeros([size, self.inputlen], dtype=np.float32)
		outputs = np.zeros([size, self.outputlen], dtype=np.float32)
		index = 0
		while 1:
			for key in self.claimType:
				l = len(self.positiveTraining[key])
				r = random.randint(0, l-1)
				inputs[index] = self.positiveTraining[key][r]
				outputs[index] = self.claimType[key]
				index = index + 1
				if index >= size:
					return {"inputs": inputs, "outputs": outputs}

	def randomBatch(self, size):
		
		inputs = np.zeros([size, self.inputlen], dtype=np.float32)
		outputs = np.zeros([size, self.outputlen], dtype=np.float32)
		index = 0
		l = len(self.training)
		for i in range(size):
			r = random.randint(0, l-1)
			inputs[i] = self.training[r]
			outputs[i] = self.trainingLabels[r]
		return {"inputs": inputs, "outputs": outputs}
	
	def versusBatch(self, size, key):
		if size % 2 != 0:
			size = size + 1
		inputs = np.zeros([size, self.inputlen], dtype=np.float32)
		outputs = np.zeros([size, 1], dtype=np.float32)
		index = 0
		pl = len(self.positiveTraining[key])
		nl = len(self.negativeTraining[key])
		for i in range(int(size/2)):
			nr = random.randint(0, nl-1)
			pr = random.randint(0, pl-1)

			inputs[index] = self.negativeTraining[key][nr]
			#inputs[index][0] = 0
			outputs[index] = [0]

			inputs[index+1] = self.positiveTraining[key][pr]
			#inputs[index+1][0] = 1
			outputs[index+1] = [1]

			index = index + 2
		return {"inputs": inputs, "outputs": outputs}
	
	def resultFromProbs(self, outputs):
		maxProb = -1 
		index = 0
		for key in self.probs:
			p = outputs[index]# * self.probs[key]
			if p >= maxProb:
				maxProb = p
				result = index
			index = index + 1
		return result
				

		
