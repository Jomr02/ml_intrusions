from DecisionTree.Decision_Tree import DecisionTree
from KMeans_DBSCAN.KMeans_DBSCAN import KMeans_DBSCAN
from Naive_Bayes.Naive_Bayes import Naive_Bayes
from Neural_Network.Red_neuronal import Neural_Network





def mostrar_menu():
    print("\nMachine Learning Menu")
    print("1. KMEAN & DBSCAN")
    print("2. Naive Bayes")
    print("3. Decision Tree")
    print("4. Neuronal Network")
    print("5. Exit")

def ejecutar_kmean():
    print("Executing KMEANS y DBSCAN...")
    KMeans_DBSCAN()

def ejecutar_naive_bayes():
    print("Executing Naive Bayes...")
    Naive_Bayes()

def ejecutar_arbol_decision():
    print("Executing Decision Tree...")
    DecisionTree()
    
def ejecutar_red_neuronal():
    print("Executing Neural Network...")
    Neural_Network()

def main():
    while True:
        mostrar_menu()
        opcion = input("Select an algorithm ")
        
        if opcion == '1':
            ejecutar_kmean()
        elif opcion == '2':
            ejecutar_naive_bayes()
        elif opcion == '3':
            ejecutar_arbol_decision()
        elif opcion == '4':
            ejecutar_red_neuronal()
        elif opcion == '5':
            print("Exiting...")
            break
        else:
            print("Please, select a valid option")




if __name__ == "__main__":
    main()