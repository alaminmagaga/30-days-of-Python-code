person = {
    'first_name': 'Asabeneh',
    'last_name': 'Yetayeh',
    'age': 250,
    'country': 'Finland',
    'is_married': True,
    'skills': ['JavaScript', 'React', 'Node', 'MongoDB', 'Python'],
    'address': {
        'street': 'Space street',
        'zipcode': '02210'
    }
}

if 'skills' in person:
    num_of_skills = len(person['skills'])
    if num_of_skills % 2 == 0:
        middle_index = int(num_of_skills / 2) - 1
        middle_skill = person['skills'][middle_index]
        print(f"The middle skill in the list is: {middle_skill}")
    else:
        middle_index = int(num_of_skills // 2)
        middle_skill = person['skills'][middle_index]
        print(f"The middle skill in the list is: {middle_skill}")

if 'skills' in person and 'Python' in person['skills']:
    print("The person has 'Python' skill.")
else:
    print("The person does not have 'Python' skill.")

if set(person['skills']) == {'JavaScript', 'React'}:
    print("He is a front-end developer.")
elif set(person['skills']) == {'Node', 'Python', 'MongoDB'}:
    print("He is a back-end developer.")
elif set(person['skills']) == {'React', 'Node', 'MongoDB'}:
    print("He is a full-stack developer.")
else:
    print("Unknown title.")

if person['is_married'] and person['country'] == 'Finland':
    print(f"{person['first_name']} {person['last_name']} lives in {person['country']}. He is married.")

