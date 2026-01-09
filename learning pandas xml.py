import pandas # Documentation https://pandas.pydata.org/docs/
import lxml


#Reads an xml file
df = pandas.read_xml("C:/LTM SMP/Stock market predictor/test.xml", xpath="//note")
# xpath: https://www.w3schools.com/xml/xpath_syntax.asp
print(df.head())