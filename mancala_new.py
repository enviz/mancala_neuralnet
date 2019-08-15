import random
import itertools
#Let's prepare the training samples
#We generate pit states randomly which can then be fed to a multinomial logistic classifier
#The objective here is to find out the best pit for a given input pit state: X_final
#The best pit which maximizes the gain will be stored in Y_final
def play(pits,pit_num):  
       
    pit = pit_num - 1
    next_pit = (pit + pits[pit]+1) % 14
    num_iter = pits[pit]
    pits[pit] = 0
    for i in range(pit+1, pit+num_iter+1):
        pits[i%14] += 1
        
    
    if pits[next_pit]==0:
        global score
        score = pits[(next_pit+1) % 14]
        pits[(next_pit+1) % 14] = 0
        #print(score)
    if pits[next_pit] > 0:
            
        play(pits,next_pit+1)
      
            
        
    return score,pits


def one_hot_vector(arr):
    vect = [0 for i in range(14)]
    indexes = [arr[0]-1]
    replacements = [1]
    for (index, replacement) in zip(indexes, replacements):
        vect[index] = replacement
    return vect



X = []
Y = []

pit_state = [5 for i in range(14)] #initial_pit state

itr = 100000               #The value of itr can be changed as per your wish
                           #NOTE: The number of iterations are not equal to number of games played. 
                           #The iteration may end in between the game,however the corresponding pit states will be saved.
                           #Just to explain that itr!=len(X_final) or len(Y_final)
while(itr>0):
    gains = []
    #returns pit_state after some pit is chosen randomly 
    #between pit 1 and pit 7
    pit_state = play(pit_state , random.randint(1,6)+1)[1]
    #print(pit_state)
    choices = [i+7 for i, value in enumerate(pit_state[7:14]) if value != 0 ] 
    #print(choices)
    if (choices==[]):
        pit_state = play([5 for i in range(14)],random.randint(1,6)+1)[1]
        choices = [i+7 for i, value in enumerate(pit_state[7:14]) if value != 0 ] 
    
    
    temp = [list.copy(pit_state) for i in range(len(choices))] 
    #print(temp)
    
    
    X.append(list.copy(temp[0]))
    
    
    
    for i in range(len(temp)):
        
        gain,state = play(temp[i],choices[i]+1)
        gains.append(gain)
        

    
    max_index = gains.index(max(gains))
    #print('gains:',gains)
    #print('max_index:',max_index)
    #print('choices:',choices)
    Y.append([choices[max_index]+1])
    #Remove the comment line to verify whether the loop is working fine.
    #print('Initial:',X[-1])
    #print('best_choice:',Y[-1])
    #print('gain:',max(gains))
    #print('pit_state:',temp[max_index])
    
    #pit state after picking the pit with max gain
    pit_state = temp[max_index]
    
    #if either the first 7 or last 7 pits gets empty,then the game ends and the pit is reset to the inital state
    #this is where the next game restarts and the number of games played is specified by the number of iterations given
    if((all(v == 0 for v in pit_state[0:8]) is True) or (all(v == 0 for v in pit_state[7:14]) is True)):
        pit_state = [5 for i in range(14)]
        continue
    itr-=1     
    
#Appending the best pit with each pit_state
for i in range(len(X)):
    X[i].append(Y[i])  
#But there maybe duplicates,hence we remove them.
#For training,only unique pit states should be present
#Here we remove the duplicates in the array of sample. Since they are generated randomly,some pit states may repeat
X.sort()
X_Y = list(num for num,_ in itertools.groupby(X))

X_final = []
Y_final = []
for i in range(len(X_Y)):
    Y_final.append(one_hot_vector(X_Y[i][-1]))
    X_final.append(X_Y[i][0:14])
sample_size = len(X_final)
print('Sample_size=',sample_size)

