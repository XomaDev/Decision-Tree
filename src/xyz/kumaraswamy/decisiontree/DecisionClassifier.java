package xyz.kumaraswamy.decisiontree;

import java.util.*;

import static java.lang.System.out;

public class DecisionClassifier {

  private static final boolean TEST = true;

  private final Object[][] trainingData;
  private final String[] propertyHeaders;

  public DecisionClassifier(Object[][] trainingData, String[] propertyHeaders) {
    this.trainingData = trainingData;
    this.propertyHeaders = propertyHeaders;

    if (TEST) {
      var question = new Question(0, "Yellow");
      // Is Color == Yellow?
      out.println(question);
      // true
      out.println(question.match("Yellow", 2, "Apple"));

      // impurity 0
      out.println("Impurity " + gini(new Object[][]{
              { "Yellow", 2, "Apple" },
              { "Yellow", 3, "Apple" }
      }));
      // impurity .5
      out.println("Impurity " + gini(new Object[][]{
              { "Yellow", 2, "Apple" },
              { "Yellow", 3, "Mango" }
      }));


      // partition the data based upon the green
      // labels
      var partitioned = partition(trainingData,
              new Question(0, "Green"));
      float impurity = gini(trainingData);

      float infoGain = findInfoGain(partitioned.first, partitioned.second, impurity);
      out.println("info gain = " + infoGain);

      findBestSplit(trainingData);
    }
    var tree = buildClassifier(trainingData);
  }

  class Question {
    private final int column;
    private final Object value;

    public Question(int column, Object value) {
      this.column = column;
      this.value = value;
    }

    public boolean match(Object... another) {
      var value = another[column];

      if (value instanceof Number number)
        return number.longValue() >= ((Number) this.value).longValue();
      return value.equals(this.value);
    }

    @Override
    public String toString() {
      var operator = value instanceof Number
              ? ">="
              : "==";
      return String.format(
              "Is %s %s %s?",
              propertyHeaders[column],
              operator,
              value);
    }
  }

  static class Pair<T> {
    T first;
    T second;

    public Pair(T first, T second) {
      this.first = first;
      this.second = second;
    }
  }

  static class BestQuestion {
    float gain;
    Question question;

    public BestQuestion(float gain, Question question) {
      this.gain = gain;
      this.question = question;
    }
  }


  /**
   * Divides the data set based upon the question,
   * asks for each of the rows to see if they
   * match, then they are segregated into trues and false
   *
   * @return Pair<True Rows, False Rows>
   */

  public Pair<Object[][]> partition(Object[][] rows, Question question) {
    List<Object[]> trueQuestions = new ArrayList<>(), falseQuestions = new ArrayList<>();

    for (Object[] row : rows) {
      if (question.match(row)) {
        trueQuestions.add(row);
      } else falseQuestions.add(row);
    }
    return new Pair<>(trueQuestions.toArray(new Object[0][]),
            falseQuestions.toArray(new Object[0][]));
  }

  /**
   * Calculates the frequencies of the labels
   * in the training data set
   */

  public static Map<String, Integer> frequencies(Object[]... rows) {
    Map<String, Integer> frequencies = new HashMap<>();

    for (Object[] row : rows) {
      var label = (String) row[row.length - 1];
      frequencies.put(label, frequencies.getOrDefault(label, 0) + 1);
    }
    return frequencies;
  }

  public float gini(Object[][] rows) {
    var frequencies = frequencies(rows);

    float impurity = 1;
    float numOfRows = rows.length;

    for (var entrySet : frequencies.entrySet()) {
      int frequency = entrySet.getValue();
      float probability = frequency / numOfRows;

      // impurity -= probability(row's frequency / numOfRows) ^ 2
      impurity -= probability * probability;
    }
    return impurity;
  }

  public float findInfoGain(Object[][] left,
                            Object[][] right,
                            float currentImpurity) {
    float totalSize = left.length + right.length;

    float avgImpurity =
            (left.length / totalSize) * gini(left) +
                    (right.length / totalSize) * gini(right);
    return currentImpurity - avgImpurity;
  }

  public BestQuestion findBestSplit(Object[][] rows) {
    /**
     * Yellow   2   Apple
     * Yellow   3   Mango
     * Orange   4   Grape
     */
    var width = rows[0].length - 1;
    var len = rows.length;

    var currentGni = gini(rows);

    var ref = new Object() {
      int i = 0;
      Question bestQuestion = null;
      float bestInfoGain;
    };

    for (; ref.i < width; ref.i++) {
      var elements = new ArrayList<>(len);
      for (Object[] row : rows)
        elements.add(row[ref.i]);

      elements.stream().distinct().forEach(o -> {
        var question = new Question(ref.i, o);
        var partitioned = partition(rows, question);

        var infoGain = findInfoGain(partitioned.first,
                partitioned.second,
                currentGni);
        if (infoGain >= ref.bestInfoGain) {
          // >= and not just > because, it's
          // just good to pick the first
          // (there maybe props with same info gains)
          ref.bestInfoGain = infoGain;
          ref.bestQuestion = question;
        }
      });
    }

    return new BestQuestion(ref.bestInfoGain, ref.bestQuestion);
  }

  static class Leaf extends DecisionTree {

    final Possibilities possibilities;

    Leaf(Object[][] rows) {
      super(null, null, null);
      possibilities = new Possibilities(rows);
    }
  }

  public static class DecisionTree {

    final Question question;
    final DecisionTree trueWay;
    final DecisionTree falseWay;

    DecisionTree(Question question,
                 DecisionTree trueWay,
                 DecisionTree falseWay) {
      this.question = question;
      this.trueWay = trueWay;
      this.falseWay = falseWay;
    }

    public Possibilities classify(Object... features) {
      return DecisionClassifier.classify(this, features);
    }
  }

  public static class Possibilities {

    final Map<String, Integer> possibilities;

    Possibilities(Object[][] rows) {
      possibilities = frequencies(rows);
    }

    public void print() {
      out.println();

      out.println("|---------------------------|");
      out.println("| Element \t|\tPossibility |");
      out.println("|-----------|---------------|");
      float size = possibilities.size();

      for (var entry : possibilities.entrySet()) {
        out.println("| " + entry.getKey() + "\t\t|\t\t" + entry.getValue() / size);
      }
    }
  }

  public DecisionTree train() {
    return buildClassifier(trainingData);
  }

  private DecisionTree buildClassifier(Object[][] rows) {
    var question = findBestSplit(rows);

    if (question.gain == 0)
      // no further, the best decision
      // gave up 0 gain
      return new Leaf(rows);
    // a pair, which has left and right
    var partition = partition(rows, question.question);

    var trueWay = buildClassifier(partition.first);
    var falseWay = buildClassifier(partition.second);

    return new DecisionTree(question.question, trueWay, falseWay);
  }

  public static Possibilities classify(DecisionTree tree, Object... features) {
    if (tree instanceof Leaf leaf)
      return leaf.possibilities;
    var question = tree.question;

    if (question.match(features))
      return classify(tree.trueWay, features);
    return classify(tree.falseWay, features);
  }
}
