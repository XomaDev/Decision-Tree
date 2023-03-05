import xyz.kumaraswamy.decisiontree.DecisionClassifier;

import static java.lang.System.out;

public class Main {

  public static void main(String[] args) {
    out.println(new Object().hashCode());
    var trainingData = new Object[][] {
             { "Green",     3,      "Apple"      },
             { "Yellow",    3,      "Apple"      },
             { "Orange",    4,      "Mango"      },
             { "Red",       1,      "Grape"      },
             { "Red",       1,      "Grape"      },
             { "Yellow",    3,      "Lemon"      },
    };
    var propertyHeaders = new String[] {
            "Color", "Diameter", "Label"
    };

    var classifier = new DecisionClassifier(trainingData, propertyHeaders)
            .train();
    out.println("\nClassify by features = \"Green\", 4");
    classifier
            .classify("Yellow", 3)
            .print();

    out.println("\nClassify by features = \"Orange\", 5");
    classifier
            .classify("Orange", 5)
            .print();
  }
}