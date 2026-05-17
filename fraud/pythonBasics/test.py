def  add(a,b):
    return a + b

def printName(name):
    print(f"Hello {name}")


class MyClass:

    def __init__(self,celcius):
        self.celcius = celcius

    @staticmethod
    def staticMethod(cc):
        print("This is a static method"+ str(cc-10))

    @classmethod
    def classMethod(cls, cc):
        print("This is a class method"+ str(cc-20))    

if __name__ == "__main__":
    print("Hello World")
    print(add(2, 3))
    printName("Alice")
    MyClass.staticMethod(20)
    MyClass.classMethod(30)


    