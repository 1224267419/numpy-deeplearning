# TODO  高亮注释
def caidan():
    print('1.新建名片 \n'
          '2.显示全部 \n'
          '3.查询名片\n\n'
          '0.退出系统\n')
    print('*' *50)
_list=[]
_dic={}
def new():
    name=input('输入姓名')
    email=input('输入email')
    phone_num= input('输入电话号码')
    qq= input('输入qq')
    _dic={
        'name':name,
        'phone':phone_num,
        'qq':qq,
        'email':email
    }
    _list.append(_dic)
    print(_list)
    print('添加成功')
def show():
    if len(_list)==0:
        print('没数据呢')
    else:
        print('姓名\t\t电话\t\tQQ\t\t邮箱\t\t')

        for i in _list:
           print('%s\t\t%s\t\t%s\t\t%s\t\t' %(i['name'],
                                              i['phone'],
                                              i['qq'],
                                              i['email']))


    print('-'*50)

def _del(j):
    _list.remove(j)
def fix(j):

    j['name'] = input('输入姓名')
    j['email']  = input('输入email')
    j['phone'] = input('输入电话号码')
    j['qq'] = input('输入qq')

    print(_list)
    print('修改成功')
    # TODO 想要改啥就改啥的话用input指定字典对应部分也可以，懒了，也可以再加一个判断函数，不输入就给原值
def cheak():
    x=input('输入名字')
    for i in _list:
        if x==i['name']:
            print('found')
            print('姓名\t\t电话\t\tQQ\t\t邮箱\t\t')
            print('%s\t\t%s\t\t%s\t\t%s\t\t' % (i['name'],
                                                i['phone'],
                                                i['qq'],
                                                i['email']))
            print('1.修改名片 \n'
                  '2.删除名片 \n\n'
                  
                  '输入其他任意键：返回主菜单\n')
            j=input('输入操作')
            if j =='1':
                fix(i)
                return
            if  j=='2':
                _del(i)
                return
            if j=='0':
                return
    else:
        print('没找到')