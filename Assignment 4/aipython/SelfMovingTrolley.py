'''
Created on 4/10/2021

@author: yolande Pretorius
This module create a strips domain for a self moving shopping trolley.

'''



if __name__ == '__main__':
    pass

shoppingAisles = {'entrance','fruit&Veg','milk&Bread','meat','checkOut'}

shopFruitList = ['bananas','apples','carrots','tomatoes','potatoes','onions','avocado']# shops stock
shopMilkBreadList =['milk','bread'] #shop stock
shopMeat =['chicken','beef','fish','pork','lamb']#shop stock

# ShopList = customerShoppingList # create a complete sholpping list
actions = set(())
LinksBetweenAislesNames =[]
DomainDictionary = {'shoppingAisles':shoppingAisles}
stripsLocations ={}

LocationItmesDictionary={'fruit&Veg':shopFruitList,'milk&Bread':shopMilkBreadList,'meat':shopMeat} # links the shopping items that user wants to the location of item in shop

from stripsProblem import Strips, STRIPS_domain
boolean = {True, False}

'''
create the name per location that the trolly can move to and from
'''
def createAislesStripsLinksName(i,j):
    name = i + "_" + j
    # print(name)
    # print(name2)
    if(name not in LinksBetweenAislesNames):
        LinksBetweenAislesNames.append(name)
        
'''
create strips between the aisles of the shop so that trolley can move from one place to the next.
'''
def createLocationStripsName(): 
    for i in shoppingAisles:
        for j in shoppingAisles:
            if(i!=j):
                createAislesStripsLinksName(i,j)
'''
Go through shopping list and add all items to the domain dictionary to create the problem
'''                
def addShoppingToDomaindictionary(customerShoppingList):
    for item in customerShoppingList:
        key = item
        if key not in DomainDictionary:
            DomainDictionary.update({item:boolean})
        
'''
go through the list of names created of areas linked together and create strips to move from one area to the next
Add the strips to the action set 
'''
def createlocationStrips(actions):   
    for item in LinksBetweenAislesNames:
        name = item
        preconditionArea = item.split("_")
        action = Strips(name,{'Aisle':preconditionArea[0]},{'Aisle':preconditionArea[1]})
        actions.add(action)
    return actions

'''
creates a dictionary to link the item from grocery lists to the area in store where item can be found
'''
def itemFromAreaDictionary(listItems,actions):
    for item in listItems:
        name = 'get'+ item
        for key,value in LocationItmesDictionary.items():
            if item in value:
                location = key
                action = Strips(name,{'Aisle':location,item:False},{'Aisle':location,item:True})
                actions.add(action)
    return actions
         
      
    
'''
=================================================main===========================================================
'''
def createShoppingDomain(customerShoppingList):
    actions = set(())  # create a new set of actions for each shopping trolley depending on shopping list
    addShoppingToDomaindictionary(customerShoppingList)
    createLocationStripsName()
    actions = createlocationStrips(actions)
    actions = itemFromAreaDictionary(customerShoppingList,actions)
    Shopping_Domain = STRIPS_domain(DomainDictionary,actions)
    
    return(Shopping_Domain)