from firebase import firebase

firebase = firebase.FirebaseApplication('https://fypdatabase-8f31d.firebaseio.com/')

x = 32

delete = firebase.delete('/user', None)
print (delete)

result = firebase.post('/user', {'First':x})
print (result)
