from constraint import *


# Defining the problem
problem = Problem()
breed = ['Boxer', 'Colie', 'Shepherd','Terrier']
favorite_game = ['Plank', 'Poles', 'Tire', 'Tunnel']
dog_names = ['Beany', 'Cheetah', 'Thor', 'Suzie']
criteria = breed + favorite_game + dog_names
problem.addVariables(criteria, [1,2,3,4])

# Defining constraint for the same initial letter of name and breed for the winning dog
def same_name_constraint(cheetah, beany, suzie, thor, terrier, boxer, colie, shepherd):
    if(cheetah == colie == 1 and beany != boxer and thor != terrier and suzie != shepherd):
        return True
    elif(beany == boxer == 1 and cheetah != colie and thor != terrier and suzie != shepherd):
        return True
    elif(thor == terrier == 1 and cheetah != colie and beany != boxer and suzie != shepherd):
        return True
    elif(suzie == shepherd == 1 and cheetah != colie and beany != boxer and thor != terrier):
        return True


# Defining the domain of the variables
problem.addConstraint(AllDifferentConstraint(), breed)
problem.addConstraint(AllDifferentConstraint(), favorite_game)
problem.addConstraint(AllDifferentConstraint(), dog_names)

# Defining the constraints
problem.addConstraint(same_name_constraint, ['Cheetah', 'Beany', 'Suzie', 'Thor', 'Terrier', 'Boxer', 'Colie', 'Shepherd'])
problem.addConstraint(lambda boxer, shepherd: boxer - shepherd == 1, ['Boxer', 'Shepherd'])
problem.addConstraint(lambda boxer, tunnel, tire: boxer != tunnel and boxer != tire, ['Boxer', 'Tunnel', 'Tire'])
problem.addConstraint(lambda shepherd, tunnel, tire: shepherd != tunnel and shepherd != tire, ['Shepherd', 'Tunnel', 'Tire'])
problem.addConstraint(InSetConstraint([3,1]), ['Poles', 'Cheetah'])
problem.addConstraint(NotInSetConstraint([2]), ['Thor'])
problem.addConstraint(lambda thor, plank: thor != plank, ['Thor', 'Plank'])
problem.addConstraint(lambda cheetah, tunnel: cheetah == tunnel or cheetah == 4, ['Cheetah', 'Tunnel'])
problem.addConstraint(lambda plank, poles: plank - poles == 1, ['Plank', 'Poles'])
problem.addConstraint(lambda suzie, shepherd: suzie != shepherd, ['Suzie', 'Shepherd'])
problem.addConstraint(lambda beany, tunnel: beany != tunnel, ['Beany', 'Tunnel'])

# Finding the solutions
solution = problem.getSolutions()

# Extracting the dictionary that we get as solution from the above statement
solution = solution[0]

# Sorting the dictionary according to the value
sorted_solution = sorted(solution.items(), key = lambda x: x[1])

# Printing the dog name, breed, ranking and sport
for i in range(0, len(sorted_solution)-2, 3):
    dog_specs = sorted_solution[i:i+3]
    rank = dog_specs[0][1]
    for item in dog_specs:
        if item[0] in dog_names:
            name = item[0]
        elif item[0] in breed:
            dog_breed = item[0]
        elif item[0] in favorite_game:
            sport = item[0]
    print('Dog with name {} belonging to breed {} likes to play {} and was ranked {}'.format(name, dog_breed, sport, rank))