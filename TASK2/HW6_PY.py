import sys
from pyspark.mllib.recommendation import ALS, Rating, MatrixFactorizationModel
from pyspark import SparkConf, SparkContext

#for SparkConf() check out http://spark.apache.org/docs/latest/configuration.html
conf = (SparkConf()
         .setMaster("local")
         .setAppName("HW6_PY")
         .set("spark.executor.memory", "1g"))
sc = SparkContext(conf = conf)


rawUserArtistData = sc.textFile("s3://csdshw/spark-emr/user_artist_data.txt")
rawArtistData = sc.textFile("s3://csdshw/spark-emr/artist_data.txt")


def solve(line):
    if '\t' in line:
        identity, name = line.split('\t',1)
        try:
            ans = (int(identity), name.strip())
            return ans
        except ValueError, e:
            return None
    else:
        return None

artistByID = rawArtistData.map(lambda line: solve(line)).filter(lambda line: line!=None)
rawArtistAlias = sc.textFile("s3://csdshw/spark-emr/artist_alias.txt")


def solve2(line):
    tokens = line.split('\t')
    if tokens[0] == None:
        return None
    else:
        try:
            ans = (int(tokens[0]), int(tokens[1]))
            return ans
        except ValueError:
            return None
			
artistAlias = rawArtistAlias.map(solve2).filter(lambda line: line!=None).collectAsMap()

bArtistAlias = sc.broadcast(artistAlias)
def solve3(line):
    userID, artistID, count = map(int, line.split(' '))
    ArtistAliasDict = bArtistAlias.value
    if artistID in ArtistAliasDict:
        finalArtistID = ArtistAliasDict[artistID]
    else:
        finalArtistID = artistID
    return Rating(userID, artistID, count)

trainData = rawUserArtistData.map(lambda line: solve3(line)).cache()
model = ALS.trainImplicit(trainData, 10, 5, alpha=1.0)

rawArtistsForUser=rawUserArtistData.map(lambda line: line.split(' ')).filter(lambda line: int(line[0]) == 2093760)
existingProducts =set(rawArtistsForUser.map(lambda line: int(line[1])).collect())

recommendations = model.call("recommendProducts", 2093760, 10)
recommendedProducts = set([i[1] for i in recommendations])
artist_recommend = artistByID.filter(lambda line: line[0] in recommendedProducts).collect()
for i in artist_recommend:
    print(i[1])

sc.stop()











