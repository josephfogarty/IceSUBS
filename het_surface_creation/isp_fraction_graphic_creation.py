# A code to get a starting initial air temperature for an LES simulation

# import needed modules
import numpy as np
import matplotlib.pyplot as plt
import os


# load path for the templates
lp = os.path.join("ice_maps","with_ponds")

#empty list of  dicst
ice_list = []
sea_list = []
pond_list = []

pond = True

#load data from files
for map_file in os.listdir(lp):
    
    # only work with .out files
    if map_file.endswith(".out"):
        
        # print the name
        print(map_file)
        
        # load the array
        arr = np.loadtxt(os.path.join(lp,map_file))
        
        # get the array
        unique, counts = np.unique(arr, return_counts=True)
        per_dict = dict(zip(unique, counts*100/arr.size))
        ice_list.append(per_dict[1.0])
        sea_list.append(per_dict[2.0])  
        if 3.0 in per_dict.keys():
            pond_list.append(per_dict[3.0])
        else:
            pond_list.append(0.0)

#colors: 'xkcd:off white', 'xkcd:midnight blue', 'xkcd:cyan'

sites = np.arange(6)
width = 0.35

if pond == False:
        
    # sort
    sea_list, ice_list = zip(*sorted(zip(sea_list,ice_list)))
        
    #create the bar chart
    p1 = plt.bar(sites, ice_list, width, edgecolor='black', color = 'xkcd:off white')
    p2 = plt.bar(sites, sea_list, width, bottom=ice_list,
                 edgecolor='black', color = 'xkcd:midnight blue')
    
    plt.ylabel('% of Domain')
    plt.title('Ice/Sea Fraction')
    plt.xticks(sites, ('S1', 'S2', 'S3', 'S4', 'S5', 'S6'))
    #plt.yticks(np.arange(0, 81, 10))
    plt.legend((p1[0], p2[0]), ('Ice', 'Sea'))
    plt.show()

if pond == True:
    
    # sort
    tuple_list = list(zip(sea_list,ice_list,pond_list))
    #resultSet = list(tuple_list)
    list_of_lists  = list(zip(*tuple_list)) 
    
    ice_list2 = list(list_of_lists[0])
    pond_list2 = list(list_of_lists[2])
    sea_list2 = list(list_of_lists[1])
    
    # Create brown bars
    plt.bar(sites, ice_list2, color='#7f6d5f', edgecolor='black', width=width)
    # Create green bars (middle), on top of the firs ones
    plt.bar(sites, pond_list2, bottom=ice_list2, color='#557f2d', edgecolor='black', width=width)
    # Create green bars (top)
    plt.bar(sites, sea_list2, bottom=pond_list2, color='#2d7f5e', edgecolor='black', width=width)
     
    # Custom X axis
    plt.xticks(sites, ('S1', 'S2', 'S3', 'S4', 'S5', 'S6'))
    plt.xlabel("Site")
     
    
    #create the bar chart
#    p1 = plt.bar(sites, ice_list2, width, edgecolor='black', color = 'xkcd:off white')
#    p2 = plt.bar(sites, pond_list2, width, edgecolor='black',
#                 color='xkcd:cyan', bottom=ice_list2)
#    p3 = plt.bar(sites, sea_list2, width, bottom=pond_list2,
#                 edgecolor='black', color = 'xkcd:midnight blue')
    
    plt.ylabel('% of Domain')
    plt.title('Ice/Sea/Pond Fraction')
    plt.xticks(sites, ('S1', 'S2', 'S3', 'S4', 'S5', 'S6'))
    #plt.yticks(np.arange(0, 81, 10))
    plt.legend(('Ice', 'Sea', 'Pond'))
    plt.show()

