# Program: CPTS 315 - Apriori Algorithm
from apyori import apriori
import pandas as pd

#Turn every line of bowsing-data.txt from a string to a list ot items
lines = []
with open("browsing-data.txt") as data:
    for line in data:
        lines.append(line.split())

#Run apriori algorithm with a max_length of 3 (provides groups of items in doubles and triples)
rules = apriori(
    lines,
    min_support=100/len(lines),
    min_confidence=0.95,
    min_lift=3,
    max_length=3)

results = list(rules)
doubles = []
triples = []
# Traverse the list of rules from the output of the apriori algorithm and
# append the qualifying sets to their designated lists (doubles and triples) with their items and confidence levels
for rule in results:
    things = [i for i in rule[0]]

    if len(things) == 2:
        doubles.append([things[0], things[1], str(rule[2][0][2])])
    elif len(things) == 3:
        triples.append([things[0], things[1], things[2], str(rule[2][0][2])])

# Throw lists (doubles and triples) into their own dataframes to manipulate the data (sorting based on confidence)
df = pd.DataFrame(doubles, columns=["item1", "item2", "confidence"])
df = df.sort_values(by='confidence', ascending=False)

df2 = pd.DataFrame(triples, columns=["item1", "item2", "item3", "confidence"])
df2 = df2.sort_values(by='confidence', ascending=False)

# Print heads of dataframes (top 5 results) into output.txt
with open("output.txt", "w+") as dataout:
    dataout.write("Output A\n")
    dataout.write(df.head().to_string())
    dataout.write("\nOutput B\n")
    dataout.write(df2.head().to_string())
    
    # Both dataframes (doubles and triples) that show all the results that fit above the 95% confidence
    # Uncomment out to see results
    # dataout.write("\n\n\n\n")
    # dataout.write(df.to_string())
    # dataout.write("\n\n\n\n")
    # dataout.write(df2.to_string())