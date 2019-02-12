res = ''
lines = open('./train.txt','r').readlines()
for line in lines:
	line = line.replace(' ','_').replace('(','').replace(')','')
	res += line
	res += '\n'
	
print(res)

