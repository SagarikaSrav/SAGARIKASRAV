#!/usr/bin/env python
# coding: utf-8

# Sorting the list from min to max age

# In[1]:


ages = [19, 22, 19, 24, 20, 25, 26, 24, 25, 24]

ages.sort()

print(ages)


# Finding Minimum age

# In[2]:


min_age=ages[0]

for i in ages:
    if i < min_age:
        min_age = i
print(min_age)


# Fining maximum age

# In[3]:


max_age=ages[0]
for j in ages:
    if j> max_age:
        max_age =j
print(max_age)


# Finding the sum of minimum and maximum ages

# In[4]:


sum_of_min_max_age= (min_age) + (max_age)

print(sum_of_min_max_age)


# Finding the median of ages

# In[5]:


from statistics import median 

median(ages)


# Finding the sum of ages

# In[6]:


sum_of_ages=sum(ages)
sum_of_ages


# Finding the length of ages

# In[7]:


len_of_ages=len(ages)

len_of_ages


# Average of ages

# In[8]:


avg_of_ages=(sum_of_ages/len_of_ages)

avg_of_ages


# Range of ages

# In[9]:


ran_of_ages=(max_age - min_age)

ran_of_ages


# Creating a dictionary dog

# In[13]:


dog={}

print(dog)


# Adding keys and values to the dictionary dog

# In[14]:


dog = {'name': 'Jin', 'color': 'White', 'breed': 'Samoyed', 'legs': 4, 'age': 5 }

dog


# Creating the dictionary student and adding keys and values

# In[47]:


student ={'first_name': 'sagarika', 'last_name': 'Mennekanti', 'gender': 'F', 'age': 27, 'marital_status': 'single', 'skills': ['python, Java'], 'country': 'India', 'city': 'Nalgonda', 'Address': '8128W'}

student


# Findig the length of the dictionary student

# In[48]:


print(len(student))


# Getting the values of the key: skills

# In[49]:


student.get('skills')


# Getting the data type of the key skills

# In[50]:


print(type('skills'))


# Adding values to the key skills

# In[51]:


student['skills'].append("C++")

student


# Getting all the key values of the dictionary student

# In[52]:


student.keys()


# Getting all the values of the dictionary student

# In[53]:


student.values()


# Creating new tuples

# In[15]:


sisters=('Gouthami', 'Bala', 'pinky', 'soumya')
brothers=('Thamus', 'Rahul', 'chintu', 'chaitu')


# In[16]:


print(sisters)


# In[17]:


print(brothers)


# Adding new tuples to create siblings

# In[18]:


siblings= brothers+sisters


# In[77]:


print(siblings)


# Length of siblings

# In[19]:


len('siblings')


# Creating a new tuple parents

# In[20]:


parents=('Balaswami', 'Josphine')


# Adding parents to siblings to create family members

# In[21]:


family_members=parents+siblings


# In[81]:


family_members


# In[2]:


it_companies = {'Facebook', 'Google', 'Microsoft', 'Apple', 'IBM', 'Oracle', 'Amazon'} 


# In[3]:


it_companies


# In[4]:


len(it_companies)


# In[5]:


it_companies.add("Twitter")


# In[6]:


it_companies


# In[121]:


new_companies={"Capegemini", "Salesforce"}


# In[122]:


it_companies.update(new_companies)


# In[123]:


it_companies


# In[125]:


it_companies.remove('Amazon')


# In[126]:


it_companies


# remove and discard are both built-in methods in python. If the element we want to delete is not present in the set then dicard method throws an error or exception whereas remove method gives an error or exception.

# In[150]:


A = {19, 22, 24, 20, 25, 26}

B = {19, 22, 20, 25, 26, 24, 28, 27}


# In[151]:


C=A.union(B)


# In[152]:


C


# In[153]:


D=A.intersection(B)


# In[154]:


D


# In[156]:


E=A.issubset(B)


# In[157]:


E


# In[158]:


F=A.isdisjoint(B)


# In[159]:


F


# In[160]:


A.update(B)
A


# In[161]:


B.update(A)
A


# Symmetric difference

# In[162]:


H=C-D


# In[163]:


H


# In[165]:


A.clear()


# In[166]:


A


# In[167]:


B.clear()


# In[168]:


B


# In[169]:


age = [22, 19, 24, 25, 26, 24, 25, 24]


# In[170]:


len(age)


# In[171]:


set_age=set(age)


# In[172]:


set_age


# In[173]:


len(set_age)


# The length of list is 8 whereas the lenth of set is 5

# In[9]:


import math as M
rad=30

_area_of_circle_= M.pi*pow(rad,2)

_area_of_circle_


# In[10]:


_circum_of_circle_=M.pi*rad*2


# In[11]:


_circum_of_circle_


# In[12]:


r = float(input ("Input the radius of the circle : "))


# In[13]:


Area_of_circle= M.pi*pow(r,2)


# In[14]:


Area_of_circle


# In[193]:


sentence= "I am a teacher and I love to inspire and teach people"


# In[194]:


unique_words = set(sentence.split(' '))


# In[195]:


unique_words


# In[198]:


lines="Name\tAge\tCountry\tCity\nAsabeneh\t250\tFinland\tHelsinki"


# In[199]:


print(lines)


# In[15]:


radius = 10
area = 3.14 * radius ** 2

result="radius = {radius}\narea=3.14 * radius **2\n"         "The area of a circle with radius {radius} is {area} meters square.".format(radius=10, area=3.14* radius**2)

print(result)


# In[19]:


n=int(input("Enter number of student's weight to be calculated"))
weights_in_lbs=[]
weights_in_kg=[]
for i in range(n):
    weights_in_lbs.append(int(input("weight {} \n".format(i+1))))
print(weights_in_lbs)
for i in range(len(weights_in_lbs)):
    lbs=0.45359237 #1lbs= 0.45359237kg
    temp=round(weights_in_lbs[i]*lbs,2)
    weights_in_kg.append(temp)
    temp=0
print(weights_in_kg)


# In[ ]:




