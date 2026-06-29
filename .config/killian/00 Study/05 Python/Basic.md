- Variable = A container for a value (string, integer,float,boolean)
- Typecasting int for input
```
name = input("What's your name: ");
age = int(input("How old are you: "));


print(f"Hello {name}");
print("Happy birthday");
print(f"You are {age} year old");
```
- Logical operator = evaluate multiple condition (or, and,not)
- Conditional expression = A one line shortcut for the if-else statement (ternary operator)
  X if condition else Y
```
num = int(input("Enter the num: "))
print("Positive" if num > 0 else "Negative")
weather = "Hot" if num >30 else "Cold" 
print(f"The weather today is too {weather}")
```
- string method
```
username = input("Enter your username: ")
if len(username) > 12 or username.find(" ") != -1 or not username.isalpha():
    print("Invalid username")
else:
    print(f"Hello {username}")
```
-   indexing = accessing elements of  a sequence using []
		[start:end:step]	
```
credit_number = "123-456-789"
# print(credit_number[0:4])
# print(credit_number[:4])
# start 0 end 4 step 1
# print(credit_number[5:])
# print(credit_number[-5]) get char right to left
# print(credit_number[::2]) 2step from index 0
# print(credit_number[::-1]) print reversed num
```
- format specifier = {value:flags} format a value based on what flags are inserted