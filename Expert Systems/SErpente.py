'''
Simple Expert System (from the original croatian description: Jednostavni ekspertni sustav)

Original Shell.py implemented by Viktor Berger (https://github.com/ViktorBerger) on the 14th of March, 2014.

SES translated, fixed, updated and re-implemented by Giovanni Rebouças (https://github.com/Giorox) on the 1st of June 2021.
'''
import copy


class ExpertSystem:
    def __init__(self, ruleFile):
        """
        Class constructor

        Params:
        ruleFile - str | path or filename to the rules and attributes file, called a knowledge-base file
        """
        # Path to the rules and attributes file
        self.ruleFilePath = ruleFile

        # Working memory, goal stack and list of already checked attributes whose value cannot be derived
        self.RM = {}
        self.goals = []
        self.checked_goals = []

        # Parse the knowledge base for attributes and rules
        self.parameters, self.rules = self.parse()

        # Print the knowledge base once for user-reference before handing control back to main
        self.printKnowledgeBase()

        # Start questioning the user
        self.start()

    def parse(self):
        """
        Parse a knowledge-base file to gather all the rules and attributes

        Params: None
        Returns:
        parameters - List(str) | List of parameters parsed from the knowledge-base file
        rules - List(str) | List of rules parsed from the knowledge-base file
        """
        parameters = {}
        rules = []
        try:
            with open(self.ruleFilePath, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print("Problem parsing file, file not found: " + self.ruleFilePath)
            return

        for line in lines:
            if not line or line.startswith('-') or line.startswith('#'):
                continue
            elif line.startswith('IF'):
                current = {}
                sides = line.replace('IF', '').split('THEN')
                current['LHS'] = {}

                conditions = sides[0].split('&')
                for condition in conditions:
                    HS = list(map(str.strip, condition.split('=', 1)))
                    current['LHS'][HS[0]] = [s.strip() for s in HS[1].split('|')]

                action = sides[1]
                HS = list(map(str.strip, action.split('=')))
                current['RHS'] = {HS[0]: HS[1]}
                rules.append(copy.deepcopy(current))
            else:
                splitLine = line.split('=', 1)
                parameters[splitLine[0].strip()] = [s.strip() for s in splitLine[1].split('|')]

        return parameters, rules

    def printKnowledgeBase(self):
        """
        The function prints attributes / parameters, their values and all the rules contained in the knowledge base

        Params: None
        Returns: None
        """
        print('-' * 105)
        print('|' + '\t' * 6 + 'KNOWLEDGE BASE' + '\t' * 6 + '|')
        print('-' * 105 + '\n')

        print("Attributes:")
        for attr, value in self.parameters.items():
            print(attr + " = " + " | ".join(value))

        print("\nRules:")
        for i, rule in enumerate(self.rules):
            print(str(i + 1) + ") " + self.rule_repr(rule))

        print('-' * 105 + '\n')

    def rule_repr(self, rule):
        """
        The function returns a string with a rule in an easy-to-read format, example:  IF condition THEN action

        Params:
        rule - str | String that contains one rule that was parsed from the knowledge-base file
        Returns:
        str | Returns a rule in an easy-to-read format.
        """
        LHS = []
        for attr, values in rule['LHS'].items():
            LHS.append(attr + " = " + "|".join(values))

        (RHSkey, RHSvalue) = list(rule['RHS'].items())[0]

        return "IF " + " & ".join(LHS) + " THEN " + RHSkey + " = " + RHSvalue

    def printRM(self):
        """
        Prints the working memory to terminal.

        Params: None
        Returns: None
        """
        print("Working memory")
        for r, v in self.RM.items():
            print(r, " = ", v)

    def getConflictRules(self, goal):
        """
        Returns a list of all conflicting rules, i.e. rules whose right side derives the value of a given parameter

        Params:
        goal - str | Parameter by which to look for conflicting rules
        Returns:
        ruleset - List(str) | A list of rules that conflinct with the goal parameter
        """
        ruleset = []
        for rule in self.rules:
            attribute = list(rule['RHS'].keys())[0]
            if attribute == goal:
                ruleset.append(rule)
        return ruleset

    def conflictRuleExists(self, goal):
        """
        The function checks if there is at least one rule whose right side derives the value of a given goal

        Params:
        goal - str | Parameter by which to look for conflicting rules
        Returns:
        bool | Whether there is ANY rule that derives from the desired goal
        """
        for rule in self.rules:
            attribute = list(rule['RHS'].keys())[0]
            if attribute == goal:
                return True
        return False

    def ruleWorks(self, rule):
        """
        The function checks if a given rule works. In other words, if all the attributes to the left of the rule are in working memory and have equal values

        Params:
        goal - str | Parameter by which to look for conflicting rules
        Returns:
        bool | Whether a certain rule's left attributes are all in working memory and have the proper values
        """
        conditions = rule['LHS']

        for param in conditions:
            if param in self.RM:
                if self.RM[param] not in conditions[param]:
                    return False
            else:
                return False
        return True

    def parameterInput(self, param):
        """
        Function for user input of a given parameter into working memory. Requires entry until the user enters one of the allowed values. Note: user input is only possible for parameters whose name ends with a '*'

        Params:
        param - str | Parameter by which the program will ask for a value to match into possible rules
        Returns: None
        """
        value = input("Please enter a parameter value '" + param + "' " + str(self.parameters[param + "*"]) + ": ")
        while(value not in self.parameters[param + "*"]):
            value = input()
        self.RM[param] = value

    def start(self):
        """
        Starts questioning the user in order to find correct values.

        Params: None
        Returns: None
        """
        # Form a stack initially composed of the most important goals (hypotheses) to be proven
        # Ask the user to enter the target hypothesis, which makes it so that there is a hypothesis at the top to be proved. If the stack is empty, then it is the END
        goal = input('Enter the hypothesis: ')
        self.goals.append(goal)

        # Main loop, will loop while there is atleast 1 goal in our stack
        while(len(self.goals) != 0):
            # Auxiliary control variables
            new_goal = False
            new_parameter = False

            # Store the current goal in the goal variable
            goal = self.goals[-1]

            # Create a set of conflict rules and store their number
            conflictRules = self.getConflictRules(goal)
            remainingRules = len(conflictRules)

            # If no conflicting rules are found, break the loop and notify the user
            if remainingRules == 0:
                print('Too little data')
                break

            # Prints a set of conflict rules and the state of working memory
            print('Conflict rules: ')
            for cr in conflictRules:
                print(self.rule_repr(cr))
            self.printRM()

            # In the loop, it goes through a set of conflicting rules
            # If the rule works:
            # 1) Take the current target off the top of the stack
            # 2) Store its right side in working memory
            # 3) Set the variable new_goal to True and exit the loop
            for cr in conflictRules:
                if self.ruleWorks(cr):
                    (RHSkey, RHSvalue) = list(cr['RHS'].items())[0]
                    self.RM[RHSkey] = RHSvalue
                    curr_goal = self.goals.pop()
                    print("The goal has been achieved: " + curr_goal + " = " + RHSvalue)
                    new_goal = True
                    break

            # If it is, go to the next goal
            if new_goal:
                continue

            # For each rule in a set of conflicting rules
            for cr in conflictRules:

                # If new_goal is True, exit the loop
                if new_goal:
                    break

                # Reduce the number of untested rules remaining
                remainingRules -= 1
                conditions = cr['LHS']

                # For each rule parameter currently being checked
                for param in conditions:
                    # If the parameter has already been checked and could not be performed, skip rule (do not check other parameters)
                    if param in self.checked_goals:
                        break
                    # If the current parameter is in working memory
                    if param in self.RM:
                        # The value of the parameter does not match the value in the working memory
                        if self.RM[param] not in conditions[param]:
                            break
                    else:  # The parameter is not in memory
                        # If any of the rules execute the current parameter, set it as the target
                        if self.conflictRuleExists(param):
                            self.goals.append(param)
                            new_goal = True
                            break
                        elif param + "*" in self.parameters:
                            # None of the rules derive a parameter if possible (parameter name ends with '*'), asks the user to enter parameter values
                            self.parameterInput(param)
                            new_parameter = True
                            break
                        else:
                            # The parameter cannot be derived from any of the rules and user cannot enter it
                            self.checked_goals.append(param)

                # If a new parameter is entered, check the rules again (exit loop)
                if new_parameter:
                    break

                # If no new target has been set and all rules have been checked, remove the target from the top and save it to a list of goals that cannot be achieved
                if remainingRules == 0 and not new_goal:
                    curr_goal = self.goals.pop()
                    self.checked_goals.append(curr_goal)
                    print('An unattainable goal: ' + curr_goal)

            # Print a very visible line that will indicate that the program has finished the current iteration of the loop
            print('-' * 100)


############################################################################################
#                                      MAIN PROGRAM                                        #
############################################################################################
if __name__ == "__main__":
    # This will instantiate the expert system, pass a file containing rules and attributes and start questioning
    esSnakes = ExpertSystem("SErpente.txt")

    # Print a very visible line that will indicate that the program is finished
    print('*' * 48 + "END OF WORK" + '*' * 48)
