import control_tool
def hello():
    while 1:
        control_tool.caidan()
        a = 100
        a = int(input('请输入想要的操作'))
        if a in [1, 2, 3]:
            control(a)
        elif a == 0:
            break
        else:
            continue

def control(a):
    if a ==1:
        control_tool.new()
    if a ==2:
        control_tool.show()
    if a ==3:
        control_tool.cheak()

if __name__ == '__main__':
   hello()


