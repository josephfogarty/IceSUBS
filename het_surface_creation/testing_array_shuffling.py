# -*- coding: utf-8 -*-
"""
Created on Fri Sep 20 10:56:17 2019

@author: jf38
"""
import numpy as np
import matplotlib.pyplot as plt
#from skimage.util import view_as_blocks
#
#def shuffle_tiles(arr, m, n):
#    a_= view_as_blocks(arr,(m,n)).reshape(-1,m,n)
#    # shuffle works along 1st dimension and in-place
#    np.random.shuffle(a_)
#    return a_
#
#
#a_test = np.arange(0,100).reshape(10,10)
#plt.matshow(a)
#
#a_shuff = shuffle_tiles(a,5,5)
#plt.matshow(a_shuff)

#input matrix:

50	51	52	53	54	0	1	2	3	4
60	61	62	63	64	10	11	12	13	14
70	71	72	73	74	20	21	22	23	24
80	81	82	83	84	30	31	32	33	34
90	91	92	93	94	40	41	42	43	44
55	56	57	58	59	5	6	7	8	9
65	66	67	68	69	15	16	17	18	19
75	76	77	78	79	25	26	27	28	29
85	86	87	88	89	35	36	37	38	39
95	96	97	98	99	45	46	47	48	49

50	51	52	53	54	0	1	2	3	4
60	61	62	63	64	10	11	12	13	14
70	71	72	73	74	20	21	22	23	24
80	81	82	83	84	30	31	32	33	34
90	91	92	93	94	40	41	42	43	44
55	56	57	58	59	5	6	7	8	9
65	66	67	68	69	15	16	17	18	19
75	76	77	78	79	25	26	27	28	29
85	86	87	88	89	35	36	37	38	39
95	96	97	98	99	45	46	47	48	49


0	1	2	3	4	5	6	7	8	9
10	11	12	13	14	15	16	17	18	19
20	21	22	23	24	25	26	27	28	29
30	31	32	33	34	35	36	37	38	39
40	41	42	43	44	45	46	47	48	49
50	51	52	53	54	55	56	57	58	59
60	61	62	63	64	65	66	67	68	69
70	71	72	73	74	75	76	77	78	79
80	81	82	83	84	85	86	87	88	89
90	91	92	93	94	95	96	97	98	99

									
									
									


#possible output for shuffling:
