import heapq


def conver_into_matrix(number_list,N):

    return [number_list[i*N:(i+1)*N] for i in range(N)]



def compute_manhattan_matrix(list_numbers,N):
    index_number = {}
    manhatton_distance = {}
    manhatton_goal = {}
    list_numbers = conver_into_matrix(list_numbers,N)
    for i in range(N):   
        for index,number in enumerate(list_numbers[i]):
            index_number[number] = [i,index]
            
            manhatton_goal[number] = [number//N,number%N]
            manhatton_distance[number] = abs(index_number[number][0]-manhatton_goal[number][0])+abs(index_number[number][1]-manhatton_goal[number][1])
    return index_number,manhatton_distance,manhatton_goal


def find_neighbour(index_number,N):
    x,y = index_number[0]
    N=N-1
    if x!=0 and y!=0 and x!=N and y!=N:
        return [[x-1,y],[x+1,y],[x,y-1],[x,y+1]]
    if x!=0 and y==0 and x!=N and y!=N:
        return [[x-1,y],[x+1,y]]
    if x==0 and y==0 and x!=N and y!=N:
        return [[x+1,y],[x,y+1]]
    if x==0 and y!=0 and x!=N and y!=N:
        return [[x,y+1],[x+1,y],[x,y-1]]
    
    if x!=0 and y!=0 and x==N and y==N:
        return [[x-1,y],[x,y-1]]
    if x!=0 and y!=0 and x!=N and y==N:
        return [[x-1,y],[x+1,y]]
    if x!=0 and y!=0 and x==N and y!=N:
        return [[x-1,y],[x,y-1],[x,y+1]]
    
   
    if x==0 and y!=0 and x!=N and y==N:
        return [[x+1,y],[x,y-1]]
    if x!=0 and y==0 and x==N and y!=N:
        return [[x,y+1],[x-1,y]]

    

        

def operation(selected_node,index_number):
    # print(selected_node,index_number)
    current_location = index_number[0]
    # print(current_location)
    swap_location = selected_node
    if current_location[0]+1 == swap_location[0] and current_location[1]==swap_location[1]:
        return "DOWN"
    elif current_location[0]-1 == swap_location[0] and current_location[1]==swap_location[1]:
        return "UP"
    elif current_location[0] == swap_location[0] and current_location[1]+1==swap_location[1]:
        return "RIGHT"
    elif current_location[0] == swap_location[0] and current_location[1]-1==swap_location[1]:
        return "LEFT"
    
def update_matrix(swap_operation,index_number,list_numbers):
    list_numbers = conver_into_matrix(list_numbers,N)
    list_numbers_ = [row[:] for row in list_numbers]
    
    if swap_operation=="DOWN":
        x,y = index_number[0]
        # print(x,y)
        temp = list_numbers_[x][y] 
        list_numbers_[x][y] = list_numbers_[x+1][y]
        list_numbers_[x+1][y] = temp
        return list_numbers_
    elif swap_operation =="UP":
        x,y = index_number[0]
        temp = list_numbers_[x][y] 
        list_numbers_[x][y] = list_numbers_[x-1][y]
        list_numbers_[x-1][y] = temp
        return list_numbers_
    elif swap_operation =="LEFT":
        x,y = index_number[0]
        temp = list_numbers_[x][y] 
        list_numbers_[x][y] = list_numbers_[x][y-1]
        list_numbers_[x][y-1] = temp
        return list_numbers_
    elif swap_operation =="RIGHT":
        x,y = index_number[0]
        
        temp = list_numbers_[x][y] 
        list_numbers_[x][y] = list_numbers_[x][y+1]
        list_numbers_[x][y+1] = temp
        return list_numbers_


def reconstruct_path(state,parents,moves):

    path = []
    current_state = tuple(state)
    # print(moves[current_state])
    while moves[tuple(current_state)] is not None:
        path.append(moves[tuple(current_state)])
        current_state = parents[tuple(current_state)]

    return list(reversed(path))

def a_star(start_state,N):
    open_list = []
    closed = set()
    
    index_number,manhatton_distance,manhatton_goal = compute_manhattan_matrix(start_state,N)
    h = sum(list(manhatton_distance.values()))
    g = 0
    f = h+g

    heapq.heappush(open_list,(h,g,start_state))
    # print(start_state,open_list)
    parents = {tuple(start_state):None}
    moves = {tuple(start_state):None}
    g_score = {tuple(start_state):None}

    while open_list:
        h,g,state = heapq.heappop(open_list)

        if h==0:
            return reconstruct_path(state=state,parents=parents,moves=moves)
        
        closed.add(tuple(state))
        index_number,manhatton_distance,manhatton_goal = compute_manhattan_matrix(state,N)
        # print(index_number,N)
        for next in find_neighbour(index_number=index_number,N=N):
            # print(next,index_number)
            move = operation(next,index_number)
            # print(move,index_number,state)
            next_state = sum(update_matrix(move,index_number,state),[])
            
            if tuple(next_state) in closed:
                continue

            new_g = g+1
            index_number,manhatton_distance,manhatton_goal = compute_manhattan_matrix(state,N)
            new_h = sum(list(manhatton_distance.values()))
            new_f = new_g+new_h

            
            if tuple(next_state) not in parents or new_g<g_score[tuple(next_state)]:
                parents[tuple(next_state)]=state
                g_score[tuple(next_state)]=new_g
                moves[tuple(next_state)]=move

                heapq.heappush(open_list,(new_h,new_g,next_state))
    return None

if __name__=="__main__":
    list_numbers = list(map(int,input().split()))
    N=int(input())
    path = a_star(list_numbers,N)
    print(len(path)-1)
    for p in path[:-1]:
        print(p)




