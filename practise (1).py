#!/usr/bin/env python
# coding: utf-8

# In[2]:


"""Hello"""


# In[3]:


"""
Hello
"""


# In[20]:


#1
import matplotlib.pyplot as plt
import numpy as np

x=np.random.randn(100)
y=np.random.randn(100)

plt.scatter(x, y, edgecolors="r", facecolors="none")
plt.show()



# In[26]:


#1
import matplotlib.pyplot as plt 
import numpy as np 
x = np.random.randn(50) 
y = np.random.randn(50)
plt.scatter(x, y, s=70, facecolors='none', edgecolors='g')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# In[43]:


#2
import matplotlib.pyplot as plt
import numpy as np

x=np.array([10,20,30,45,3,5,3,4,3,5,34])
y=np.array([20,30,40,63,6,3,56,4,6,46,56])
sizes=np.array([20,30,40,2,42,4,2,53,5,67,65])

plt.scatter(x, y, s=sizes)
plt.show()


# In[ ]:





# In[8]:


#3
import matplotlib.pyplot as plt
import numpy as np

math_marks = [88, 92, 80, 89, 100, 80, 60, 100, 80, 34]
science_marks = [35, 79, 79, 48, 100, 88, 32, 45, 20, 30]
marks_range = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

plt.scatter(math_marks, science_marks, marks_range
           )

plt.show()


# In[44]:


#4
dic1={1:10, 2:20}
dic2={3:30, 4:40}
dic3={5:50,6:60}

mydict={}
mydict.update(dic1)
mydict.update(dic2)
mydict.update(dic3)
print(mydict)


# In[47]:


#5
d = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60}
def present(x):
    if x in d:
        print(f"{x} is present in {d}")
    else:
        print(f"{x} is not present in {d}")
x=int(input("Enter the key you want to check: "))
present(x)


# In[57]:


#6
d={}
n=int(input("Enter the value of n: "))
for i in range(0,n):
    d.update({(i+1):pow((i+1) , 2)})
    
print(d)


    


# In[61]:


#7
d = {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60}
d.pop(1)
print(d)


# In[65]:


#8
keys = ['red', 'green', 'blue']
values = ['#FF0000','#008000', '#0000FF']
d={}
for i in range(0,3):
    d[keys[i]]=values[i]
print(d)


# In[67]:


#8
keys = ['red', 'green', 'blue']
values = ['#FF0000','#008000', '#0000FF']
d=dict(zip(keys,values))
print(d)


# In[80]:


#9
L = [{"V":"S001"}, {"V": "S002"}, {"VI": "S001"}, {"VI": "S005"}, {"VII":"S005"}, {"V":"S009"},{"VIII":"S007"}]
print("Original List: ",L)
u_value = set( val for dic in L for val in dic.values())
print("Unique Values: ",u_value)


# In[100]:


L = [{"V":"S001"}, {"V": "S002"}, {"VI": "S001"}, {"VI": "S005"}, {"VII":"S005"}, {"V":"S009"},{"VIII":"S007"}]

print(uv)
        


# In[103]:


#10
s=input("Enter the String: ")
d={}
for i in s:
    d[i]=d.get(i,0)+1
print(d)    
    


# In[118]:


# #11
s=input("Enter a Sentence: ")
# a=s.split(' ')
# s1=" "
# for i in range(len(a)):
#     s1+='-'+a[i]
# print(s1)
s=s.replace(" ","-")
print(s)


# In[124]:


#12
import numpy as np
x=np.random.randint(100,200,10)
print(x)
min=x[0]
for i in range (len(x)):
    if x[i]<min:
        min=x[i]
print(f"The smallest random number is: {min}")


# In[129]:


#13
d={"India":"Ruppee", "USA":"Dollar", "Japan":"Yen", "UK":"Pound", "Dubai":"Dirham"}

for key in d:
    print(f"{key}: {d[key]}")

    


# In[143]:


#14
import numpy as np
x=1-2j
print(type(x))
print(x)
print(x.real)
print(x.imag)
print(abs(x))
print(np.conj(x))


# In[147]:


#15
s=input("Enter the string in whoch you want ot replace Hello: ")
s=s.strip()
s=s.replace("Hello","Help")
print(s)


# In[146]:


#16
import numpy as np
x=np.random.randint(100,200,10)
print(x)
min=x[0]
for i in range (len(x)):
    if x[i]<min:
        min=x[i]
print(f"The smallest random number is: {min}")


# In[157]:


#17
s=input("Enter the String: ")
print(s[-1:]+s[1:-1]+s[:1])


# In[159]:


#18
s=input("Enter the string: ")
ss=input("Enter the substring to be searched: ")
if ss in s:
    print(f"{ss} is a substring in {s}")
else:
    print(f"{ss} is not a substring in {s}")

          


# In[222]:


#19 same as #11
import numpy as np
x=np.array([1,2,3,4])
for i in range (0,len(x)):
    print(x[i])


# In[165]:


#20
s=input("Enter the string to check: ")
s1=s[::-1]
if(s==s1):
    print(f"{s} is a palindrome")
else:
    print(f"{s} is not a palindrome")


# In[176]:


#21
s=input("Enter a sentence: ")
s=s.split(" ")
s1=""
ans=""
# print(s)
for i in range (len(s)):
    s1=s[i].capitalize()
#     print(s1)
    ans+=" "+s1
print(ans)
    
    


# In[184]:


#22
s=input("Enter the String: ")
c=input("Enter the character whose count you want: ")
count=0
for i in range (len(s)):
    if s[i]==c:
        count=(int)(count+1)
#         print(count)
print(count)


# In[186]:


#23
d={"Maharashtra":"Mumbai", "Gujurat":"Gandhinagar", "Bihar":"Patna", "UP":"Lucknow"}
print(d)
d.update({"MP":"Bhopal"})
print(d)


# In[188]:


#24
s=input("Enter the string: ")
v=0
c=0
d=0
sp=0
sc=0
for i in range (len(s)):
    if s[i]=='a' or s[i]=='A' or s[i]=='e' or s[i]=='E' or s[i]=='i' or s[i]=='I' or s[i]=='o' or s[i]=='O' or s[i]=='u' or s[i]=='U':
        v=(int)(v+1)
    elif (s[i] >= 'a' and s[i] <= 'z')  or   (s[i] >= 'A' and s[i] <= 'Z'):
        c=(int)(c+1)
    elif s[i]>='0' and s[i]<='9':
        d=(int)(d+1)
    elif s[i]==' ':
        sp=(int)(sp+1)
    else:
        sc=(int)(sc+1)
print(f"No. of digits: {d}")
print(f"No. of special characters: {sc}")
print(f"No. of vowels: {v}")
print(f"No. of consonants: {c}")
print(f"No. of white spaces: {sp}")
        


# In[3]:


#25
s=input("Enter String: (Upto 15 characters)  ")
for i in range(min(15,len(s))):
    print(f"Element : {s[i]}\tElement Number : {(i+1)}")

    


# In[198]:


#26
import numpy as np

x=np.array([1,1,1,1])
print(x)


# In[209]:


#27
import numpy as np

x=np.array([[1,2,3],[4,5,6]])
r=[]
n=int(input("Enter the number of row elements: "))
print("Enter the elements one by one")
for i in range(0,n):
    a=int(input())
    r.append(a)
if r in x:
    print("Row is present")
else:
    print("Row is absent")


# In[229]:


#28
import numpy as np
l1=[]
l2=[]
l3=[]
l4=[]
print("Enter the elements of the first row of first matrix: ")
for i in range (2):
    a=int(input())
    l1.append(a)
print("Enter the elements of the second row of first matrix: ")
for i in range (2):
    a=int(input())
    l2.append(a)
print("Enter the elements of the first row of second matrix: ")
for i in range (2):
    a=int(input())
    l3.append(a)
print("Enter the elements of the second row of second matrix: ")
for i in range (2):
    a=int(input())
    l4.append(a)
    
m1=np.array([l1,l2])
m2=np.array([l3,l4])



print(f"1st Matrix: \n {m1}")
print(f"2nd Matrix: \n {m2}")

sum=np.add(m1,m2)
print(f"The Addition is: \n{sum}")

diff=np.subtract(m2,m1)
print(f"The Difference is: \n{diff}")

mul=np.dot(m1,m2)
print(f"The Multiplication is: \n{mul}")


# 
# import numpy as np
# l=[]
# n=int(input("Enter the numer of elements in the array: "))
# print("Enter the elements of the array: ")
# for i in range(n):
#     z=int(input())
#     l.append(z)
# m=np.array(l)
# a=np.bincount(m).argmax()
# print(a)
# 

# #29
# import numpy as np
# l=[]
# n=int(input("Enter the numer of elements in the array: "))
# print("Enter the elements of the array: ")
# for i in range(n):
#     z=int(input())
#     l.append(z)
# m=np.array(l)
# a=np.bincount(m).argmax()
# print(a)
# 

# In[241]:


#29
import numpy as np
l=[]
n=int(input("Enter the numer of elements in the array: "))
print("Enter the elements of the array: ")
for i in range(n):
    z=int(input())
    l.append(z)
m=np.array(l)
a=np.bincount(m).argmax()
c=np.bincount(m).max()
print(f"The element with maximum occurences is: {a}")
print(f"It's count: {c}")


# In[243]:


#30
import numpy as np
x2=np.array([[1,2],[3,4]])
x1=x2.flatten()
print(x1)


# In[247]:


#31
import numpy as np
x2=np.array([[1,2],[3,4],[5,6]])
result=x2.sum(axis=0)
print(result)


# In[260]:


#32
import numpy as np
x=np.random.randint(0,100,20)
m=np.mean(x)
v=np.var(x)
sd=np.sqrt(v)
print(f"The array is: \n{x}")
print(f"The mean is: {m}")
print(f"The variance is {v}")
print(f"The standard deviatio is {sd}")


# In[289]:


#33
import numpy as np
x=np.array(["Python","is","easy"])
ans=np.char.join(" ",x)
print(ans)


    


# In[291]:


#34
import numpy as np
import matplotlib.pyplot as plt

x=np.array([1,2,3,4,5])
y=np.array([5,10,15,20,25])

plt.plot(x, y)
plt.show()


# In[299]:


#35
# n=input("Enter a number: ")
# print(n[::-1])
n=int(input("Enter a number: "))
rev=0
temp=(int)(n)
while(temp>0):
    rem=(int)(temp%10)
    rev=(int)(rev*10 + rem)
    temp//=10
print(rev)


# In[318]:


#36
import numpy as np
n=int(input("Enter the number: "))
for i in range(1,n+1):
    a=[]
    for j in range (1,i+1):
        print(j,end="")
        if(j<i):
            print("+",end="")
        a.append(j)
    print(f"={np.sum(a)}")
    


# In[329]:


#37
def triplets(l):
    c=(int)(0)
    m=(int)(2)
    while c < l:
        for i in range(1,m):
            a=m*m - i*i
            b=2*m*i
            c=m*m + i*i
            if(c>l):
                break
            print(f"{a} {b} {c} ")
        m=m+1
l=int(input("Enter the highest value: "))
print("The triplets are: ")
triplets(l)

        


# In[333]:


#38
n=input("Enter the binary number: ")
c0=0
c1=0
for i in range(len(n)):
    if(n[i]=='0'):
        c0+=1
    elif(n[i]=='1'):
        c1+=1
if c0==1 or c1==1:
    print("Yes")
else:
    print("No")


# In[29]:


#39
def selectionSort(array, n):
    c=0
    for i in range(n):
        max_i = i

        for j in range(i + 1, n):
            if array[j] > array[max_i]:
                max_i = j
        (array[n-1], array[max_i]) = (array[max_i], array[n-1])
        c=c+1
    print(c)


l = [-2, 45, 0, 11, -9]
n = len(l)
selectionSort(l, n)
print('Sorted Array in Ascending Order:')
print(l)

    


# In[37]:


#40
for i in range(1,7):
    for j in range(1,i+1):
        print(i,end="")
    if(i!=6):    
        print("\\n")


# In[39]:


#41
keys = ['red', 'green', 'blue']
values = ['#FF0000','#008000', '#0000FF']
d={}
# for i in range(0,3):
#     d[keys[i]]=values[i]
d=dict(zip(keys,values))
print(d)


# In[5]:


#42
n=int(input("Enter the number: "))
c=0
while n>10:
    n//=6
    c+=1
print(c)


# In[ ]:





# In[106]:





# In[119]:


#43
f=open('idk.txt', 'r')
for line in f:
    output=line.title()
    print(output)
f.close()


# In[120]:


#44
c=0
f=open('idk.txt', 'r')
s=input("Enter the character whose count you want: ")
for line in f: 
    for word in range(len(line)):
        if line[word]==s:
            c+=1
print(f"Number of times {s} appears in the file is {c}")
f.close()


# In[144]:


#45
f=open('idk.txt', 'w')
f.write("HI\nHello\nThis is sample file")
f.close()
f=open('copy.txt','a')
f.write("\n I have appended this")
f.close()

#46
f=open('idk.txt','r')
print(f.read())
f.close()
f1=open('copy.txt','r')
print(f1.read())
f1.close()


# In[142]:


#47
f1=open('idk.txt','r')
f2=open('copy.txt','w')
k=f1.read()
f1.close()
f2.write(k)
f2.close()
f=open('copy.txt','r')
print(f.read())

#contents of copy.txt after copying: 
#HI
#Hello
#This is sample file
f.close()


# In[159]:


#48
# f=open('djis.txt','r')
# print(f.read()) # will throw error as file does not exist
import os.path
print(os.path.exists('copy.txt'))
name=[]
rollno=[]
n=int(input("Enter the number of students: "))
for i in range(n):
    name.append(input(f"Enter Name of Student of {i+1}: "))
    rollno.append(input(f"Enter Roll No. of Student of {i+1}: "))
d=dict(zip(name,rollno))
s1=""
s2=""
for i in name:
    s1+=i+" "
for i in rollno:
    s2+=i+" "

    
f=open('copy.txt','a')
f.write("\nNames: ")
f.write(s1)
f.write("\nRollnos: ")
f.write(s2)
f.close()
f=open('copy.txt','r')
print(f.read())
f.close()
    



# In[163]:


#49
file1=input("Enter the source file: ")
file2=input("Enter the destination file: ")

f1=open(file1,'r')
f2=open(file2,'w')
f2.write(f1.read())
f1.close()
f2.close()
print("This is the destination file: \n")
f=open(file2,'r')
print(f.read())
f.close()


# In[191]:


#50
f=open('new.txt','r')
for line in f:
    for word in line.split():
        print(word)
        
f.close()


# In[190]:


#51
f=open('new.txt','r')
for line in f:
    for character in line:
        print(character)
f.close()


# In[209]:


#52
f=open('new.txt','r')
w=0
s=0
c=0
lines=0
for line in f:
    for word in line.split():
        w+=1
    for character in line:
        if(character==' '):
            s+=1
        elif(character=='\r'):
            pass
        else:
            c+=1
    lines+=1

print(f"Number of words: {w}")
print(f"Number of spaces: {s-1}")
print(f"Number of characters: {c-lines+1}")
print(f"Number of lines: {lines}")


    


# In[218]:


#53
class InvalidMarks(Exception):
    pass
m=int(input("Enter your marks: "))
try:
    if m<=100:
        print(f"Your marks are: {m}")
    else:
        raise InvalidMarks
except InvalidMarks:
    print("Marks are invalid \nEnter marks less than or equal to 100")
    


# In[220]:


help(Exception)


# In[4]:


#54
class DivideByZero(Exception):
    pass
a=int(input("Enter a: "))
b=int(input("Enter b: "))
c=int(input("Enter c: "))
d=int(input("Enter d: "))
try:
    if b==0 or d==0:
        raise DivideByZero
    else:
        ans=((a+d) + (b*c))/ (b*d)
        print(f"The answer is: {ans}")
except DivideByZero:
    print("You cannot divide by zero")


# In[8]:


#55
class InvalidAge(Exception):
    pass
age=int(input("Enter you age: "))
try:
    if(age>=18):
        print("Age is Valid")
    else:
        raise InvalidAge
except InvalidAge:
    print("Age is Invalid")


# In[11]:


#56
f1=input("Enter a file name: ")
try:
    f=open(f1,'r')
    print(f.read())
except FileNotFoundError:
    print("File name entered does not exist \nPls enter filename present in your folder")

    


# In[21]:


#57
class TypeError(Exception):
    pass
def check(n):
    try:
        if(type(n)==str):
            print(f"{n} is a String")
        else:
            raise TypeError
    except TypeError:
        print(TypeError)
x="Preet"
y=8
check(x)
check(y)

    


# #58
# class Complex ():
#     def __init__(self):
#         self.realPart = int(input("Enter the Real Part: "))
#         self.imgPart = int(input("Enter the Imaginary Part: "))            
# 
#     def display(self):
#         print(self.realPart,"",self.imgPart,"i", sep="")
# 
#     def sum(self, c1, c2):
#         self.realPart = c1.realPart + c2.realPart
#         self.imgPart = c1.imgPart + c2.imgPart
#         print(f"Sum of Real parts: {self.realPart}")
#         print(f"Sum of Imaginary parts: {self.imgPart}")
# 
# # c1 = Complex()
# # c2 = Complex()
# # c3 = Complex()
# 
# print("Enter first complex number")
# c1.__init__()
# print("First Complex Number: ", end="")
# c1.display()
# 
# print("Enter second complex number")
# c2.__init__()
# print("Second Complex Number: ", end="")
# c2.display()
# 
# print("Sum of two complex numbers is ", end="")
# c3.sum(c1,c2)
# 
#         
#     
#     

# In[8]:


#58
class Complex:
    def input(self):
        self.r=int(input("Enter the Real Part: "))
        self.i=int(input("Enter the Imaginary Part: "))
    def display(self):
        if(self.i<0):
            print(f"{self.r}  {self.i}i")
        else:
            print(f"{self.r} + {self.i}i")
    def calculate(self,c1,c2):
        self.r=c1.r+c2.r
        self.i=c1.i+c2.i
        print(f"Sum of real parts: {self.r}")
        print(f"Sum of Imaginary parts: {self.i}")
c1=Complex()
c2=Complex()
c3=Complex()
print("Enter the first complex number: \n")
c1.input()
print("Enter the second complex number: \n")
c2.input()
print("First number: ")
c1.display()
print("Second number: ")
c2.display()

c3.calculate(c1,c2)
print("The final complex number is: ")
c3.display()






# In[1]:


#59
class Triangle:
    def __init__(self,a,b,c):
        self.a=a
        self.b=b
        self.c=c
    def perimeter(self):
        p=a+b+c
        return p
print("Enter the sides of the traingle: ")
a=int(input())
b=int(input())
c=int(input())

o=Triangle(a,b,c)

p=o.perimeter()
print(f"Perimeter = {p}")


# In[7]:


#60
class Lists:
    def Append(self,l,n):
        l.append(n)
    def Delete(self,l,n):
        l.remove(n)
    def Display(self,l):
        print(l)
o=Lists()
thislist = ["apple", "banana", "cherry"]
o.Append(thislist,"mango")
o.Display(thislist)
o.Delete(thislist,"banana")
o.Display(thislist)
        
        


# In[13]:


#61
class Operations:
    def __init__(self,a,b):
        self.a=a
        self.b=b
    def add(self):
        s=a+b
        return s
    def sub(self):
        d=a-b
        return d
    def mul(self):
        p=a*b
        return p
    def div(self):
        q=(float)(a/b)
        return q
a=int(input("Enter First Number: "))
b=int(input("Enter Second Number: "))

o=Operations(a,b)

s=o.add()
print(f"The Addition is: {s}")
d=o.sub()
print(f"The Substration is: {d}")
p=o.mul()
print(f"The Multiplication is: {p}")
q=(float)(o.div())
print(f"The Division is: {q}")


# In[33]:


#62
class Student:
    def __init__(self,student_id,student_name):
        self.id=student_id
        self.name=student_name
    def stud_class(self,attr):
        setattr(self, attr, attr)
    def display(self):
        print(f"ID: {self.id}")
        print(f"Name: {self.name}")
        print(f"Class: {self.stud_class}")
        
sid=int(input("Enter your ID: "))
name=input("Enter your Name: ")
c=input("Enter your Class: ")
o=Student(sid,name)
setattr(Student,"stud_class",c)
o.display()


# In[30]:


#63
class Rev:
    def __init__(self,s):
        self.s=s
    def rev(self):
        s1=self.s[::-1]
        return s1
    
s=input("Enter the String: ")
s1=""
for word in s.split():
    o=Rev(word)
    s1+=o.rev()+" "
print(s1)


# In[37]:


#64
class String:
    def get_String(self):
        self.s=input("Enter the String: ")
    def print_String(self):
        print(self.s.upper())

x=String()
x.get_String()
x.print_String()


# In[39]:


#65
class Circle:
    def __init__(self,r):
        self.r=r
    def area(self):
        a=3.14*r*r
        return a
    def peri(self):
        p=2*3.14*r
        return p
r=int(input("Enter the radius: "))
x=Circle(r)
print(f"Area is : {x.area()}")
print(f"Perimeter is : {x.peri()}")
    


# In[42]:


#66
class Vehicle:
    def __init__(self,max_speed,mileage):
        self.ms=max_speed
        self.m=mileage
    def display(self):
        print(f"Max Speed is: {ms}")
        print(f"Mileage is: {m}")
class Bus(Vehicle):
    pass
ms=int(input("Enter the Maximum Speed: "))
m=int(input("Enter the Mileage: "))
o=Bus(ms,m)
o.display()


# In[87]:


#67
import pandas as pd
import matplotlib.pyplot as plt
pd.options.display.max_rows=9999
df=pd.read_csv('data.csv')
# print(df.dtypes)
# print(df.head(5))
# print(df.tail(5))
# print(df.to_numpy())
# print(df.describe())
# print(df.sort_values('Duration',ascending=True))
# print(df['Pulse'])
# print(df)
# print()
# print()

# newdf=df.dropna()
# print(newdf)
df.dropna(inplace=True)
df["Date"]=pd.to_datetime(df["Date"])
for i in df.index:
    if df.loc[i,"Duration"] > 120:
        df.loc[i,"Duration"] = 120
        
df.drop_duplicates(inplace=True)
        
print(df.corr())
df.plot()
plt.show()


# In[109]:


#67 graph

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_rows=9999
df=pd.read_csv('data2.csv')
# print(df)

df.plot()
plt.show()

df.plot(kind="scatter", x="Duration", y="Calories")
plt.show()

sns.scatterplot(x=df["Duration"],y=df["Pulse"])


df["Maxpulse"].plot(kind="hist")
plt.show()



sns.lineplot(x=df["Duration"],y=df["Pulse"])


# sns.barplot(x=df["Duration"],y=df["Pulse"])

sns.histplot(df["Duration"])
plt.show()





# In[123]:


#68
n=int(input("Enter the length of the list: "))
l=[]
for i in range(n):
    a=int(input("Enter value: "))
    l.append(a)
j=0
for i in range(n):
    if l[i]!=0:
        l[i],l[j] = l[j],l[i] 
        j+=1
print(l)
            


# In[176]:


#69
s=input("Enter the comma separated string: ")
l=s.split(",")
l1=[]
for i in range(len(l)-1,-1,-1):
    a=l[i].strip()
    l1.append(a)
# print(l1)
l1.sort()
print(",".join(l1))


# In[35]:


#70
import numpy as np
c=50
h=30
l=input("Enter comma separated values: ")
l=l.split(",")
l1=[]
for i in l:
    a=round(math.sqrt((2*c*(int)(i))/h))
    l1.append(str(a))
    
print(",".join(l1))
    


# #

# In[38]:


#71
n=int(input("Enter the length of the list: "))
l=[]
print("Enter list items: ")
for i in range(n):
    a=int(input())
    l.append(a)
ans=[]
dup=[]
for i in l:
    if i not in dup:
        ans.append(i)
        dup.append(i)
print(ans)
        


# In[40]:


#72
def printDic(d,n):
    for i in range(0,n):
        d.update({(i+1):pow((i+1) , 2)})
    print(d)
n=int(input("Enter n: "))
d={}
printDic(d,n)


# In[44]:


#73
n=int(input("Enter the number: "))
temp=n
d=0
while temp>0:
    rem=temp%10
    d=(int)(d+rem)
    temp=temp/10
print(f"Number of digits: {d}")

    


# In[62]:


#74
def Prime(n):
    for i in range(2,(int)((n/2)+1)+1):
        if n%i==0:
            return False
        else:
            return True
n=int(input("Enter n: "))
print(2)
for i in range(1,n+1):
    if(Prime(i)):
        print(i)


# In[67]:


#75
def Print(n):
    if n==0:
        print(0)
    else:
        Print(n-1)
        print(n)
n=int(input("Enter n: "))
Print(n)


# In[166]:


#76
def student_data(student_ID,**kwargs):
    print(f"Your ID:{student_ID}")

    if 'student_name' in kwargs and 'student_class' not in kwargs:
        print(f"Your Name is :{kwargs['student_name']}")

    if 'student_name' and 'student_class' in kwargs:
        print(f"Your Name is :{kwargs['student_name']}")
        print(f"Your class is :{kwargs['student_class']}")

student_data(student_ID=1221,student_name='Karna')
student_data(student_ID=122,student_name='Preet',student_class='B')


# In[75]:


#77
class Roman:
    def romantoint(self,s):
        d={'I':1, 'V':5, 'X':10, 'L':50, 'C':100, 'D':500, 'M':1000, 'IV':4, 'IX':9, 'XL':40, 'XC':90, 'CD':400, 'CM':900}
        amt=0
        i=0
        while i < len(s):
            if i+1 < len(s) and s[i:i+2] in d:
                amt+=d[s[i:i+2]]
                i+=2
            else:
                amt+=d[s[i]]
                i+=1
        return amt
                
            
            
s=input("Enter in Roman: ")
o=Roman()
print(o.romantoint(s))

                
        


# In[81]:


#78
class Sets:
    def subset(self,ss):
        return self.subsetRecur([],sorted(ss))
    def subsetRecur(self,current,ss):
        if(ss):
            return self.subsetRecur(current,ss[1:]) + self.subsetRecur(current+[ss[0]],ss[1:])
        return [current]
s=input("Enter the set with commas in between")
l=s.split(",")
o=Sets()
print(o.subset(l))


# In[84]:


#79
l=[]
n=int(input("Enter the number of values in list: "))
print("Enter values one by one: \n")
for i in range(n):
    a=int(input())
    l.append(a)
s=int(input("Enter the sum: "))
for i in range(len(l)):
    for j in range(len(l)):
        if (l[i]+l[j]) == s:
            n1=i
            n2=j
            break
print(f"{l[n1]}(Index: {n1}) + {l[n2]}(Index: {n2}) = {s}")


# In[86]:


#80
class Pow:
    def __init__(self,x,n):
        self.x=x
        self.n=n
    def power(self):
        ans=1
        for i in range(n):
            ans=ans*x
        return ans
x=int(input("Enter value of x: "))
n=int(input("Enter value of n: "))
o=Pow(x,n)
print(o.power())


# In[101]:


#81
import numpy as np

l1=[]
n=int(input("Enter length of array: "))
print("Enter temperatures in Centigrade: ")
for i in range(n):
    x=float(input())
    l1.append(x)
c=np.array(l1)
l2=[]
for i in range(n):
    x=(float)((c[i]*1.8)+32)
    print(x)
    l2.append(x)
f=np.array(l2)
print(np.round(f,2))

    
    


# In[104]:


#82
import numpy as np

l1=[]
n1=int(input("Enter length of set 1: "))
print("Enter the set values: ")
for i in range(n1):
    x=float(input())
    l1.append(x)
l1.sort()
a1=np.array(l1)
l2=[]
n2=int(input("Enter length of set 2: "))
print("Enter the set values: ")
for i in range(n2):
    x=float(input())
    l2.append(x)
l2.sort()
a2=np.array(l2)
l3=[]
for i in a1:
    if i in a1 and i in a2:
        pass
    elif i in a1:
        l3.append(i)
a3=np.array(l3)
print(a3)


# In[106]:


#83
import pandas as pd
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df=pd.DataFrame(exam_data,index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
print(df)


# In[107]:


#84
import pandas as pd
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df=pd.DataFrame(exam_data,index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
print(df.head(3))


# In[111]:


#85
import pandas as pd
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df=pd.DataFrame(exam_data,index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
print("Name: \n")
print(df['name'])
print("Scores: \n")
print(df['score'])


# In[112]:


#86
import pandas as pd
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df=pd.DataFrame(exam_data,index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
c=input("Enter the column you want to print: ")
print(df[c])
r=input("Enter the row you want to print: ")
print(df.loc[r])


# In[116]:


#87
import pandas as pd
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df=pd.DataFrame(exam_data,index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
for i in df.index:
    if df['attempts'].loc[i]>2:
        print(df.loc[i])


# In[120]:


#88
import pandas as pd
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df=pd.DataFrame(exam_data,index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
r=0
for i in df.index:
    r+=1
print(f"Number of Rows: {len(df)}")
print(f"Number of Columns: {len(df.columns)}")


# In[123]:


#89
import pandas as pd
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df=pd.DataFrame(exam_data,index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

for i in df.index:
    if df['score'].loc[i]<=25 and df['score'].loc[i]>=15:
        print(df.loc[i])


# In[125]:


#90
import pandas as pd
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df=pd.DataFrame(exam_data,index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

for i in df.index:
    if df['attempts'].loc[i]<2 and df['score'].loc[i]>15:
        print(df.loc[i])


# In[143]:


#91
import pandas as pd
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df=pd.DataFrame(exam_data,index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

df.loc['k']=['Preet',25,1,'yes']
print(df)
print()
print()
df=df.drop('k')
print(df)


# In[158]:


#92
import pandas as pd
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df=pd.DataFrame(exam_data,index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

df.sort_values(by=['name'], inplace=True, ascending=False)
print(df)
print()
print()
df.sort_values(by=['score'],inplace=True,ascending=True)
print(df)


# In[164]:


#93
import pandas as pd
exam_data = {'name': ['Anastasia', 'Dima', 'Katherine', 'James', 'Emily', 'Michael', 'Matthew', 'Laura', 'Kevin', 'Jonas'],
'score': [12.5, 9, 16.5, np.nan, 9, 20, 14.5, np.nan, 8, 19],
'attempts': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
'qualify': ['yes', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'no', 'yes']}
df=pd.DataFrame(exam_data,index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])

df['qualify']=df['qualify'].map({'yes':True, 'no': False})
print(df)


# In[71]:


#94
import pandas as pd

data = {
  "calories": [420, 380, 390],
  "duration": [50, 40, 45]
}

df = pd.DataFrame(data, index = ["day1", "day2", "day3"])
df["calories"].loc["day1"]=350
print(df) 

