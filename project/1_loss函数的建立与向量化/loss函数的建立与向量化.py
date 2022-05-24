import numpy as np


def  compute_error_for_line_given_points(b,w,points):
    total=0
    for i in range (0,len(points)):
        x=points[i,0]
        y = points[i, 1]
        total+=((y-(w*x+b)))**2
    return  total/   float(len(points))
###第一版损失函数，下一版考虑用向量进行优化而不是简单for函数

def step_gradient(b_current,w_current,points,learningRate):
    b_gradient = 0
    w_gradient = 0
    N=float(len(points))
    for i in range (0,len(points)):
        x=points[i,0]
        y=points[i,1]
        b_gradient+=-2*(y-(w_current *x )+b_current)
        w_gradient += -2 * x * (y - (w_current * x) + b_current)    #对w和b求导，得到梯度(以后可以考虑torch内置求导)
    new_b = b_current - (learningRate * b_gradient/N)               # b_hat=b-α/n∑Δb=b-α/n∑(dloss/db)
    new_w = b_current - (learningRate * w_gradient/N)               # w_hat=w-α/n∑(dloss/dw)=w-α/n∑Δw
    return [new_b,new_w]
#训练


def gradient_descent_runner(points, start_b,start_w,learning_rate,num_iterations):
    b=start_b
    w=start_w
    for i in  range(num_iterations):
        b,w=step_gradient(b,w,np.array(points),learning_rate)
    return [b,w]

def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    learning_rate = 0.0001
    initial_b = 0 # initial y-intercept guess
    initial_m = 0 # initial slope guess
    num_iterations = 1000
    print("Starting gradient descent at b = {0}, m = {1}, error = {2}"
          .format(initial_b, initial_m,
                  compute_error_for_line_given_points(initial_b, initial_m, points))
          )
    print("Running...")
    [b, m] = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print("After {0} iterations b = {1}, m = {2}, error = {3}".
          format(num_iterations, b, m,
                 compute_error_for_line_given_points(b, m, points))
          )



if __name__ == '__main__':
    run()