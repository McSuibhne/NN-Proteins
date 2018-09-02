import java.io.*;
import java.text.DateFormat;
import java.text.SimpleDateFormat;
import java.util.*;
import java.util.Map.Entry;

//Reads in a prespecified set of raw protein datasets and generates the statistical
// analysis files based on the generated training and testing sets.
public class GenerateStats {

    static  int FOLD_START = 0;
    static  int FOLD_RANGE = 1;
    static final int WINDOW_SIZE = 5;
    static final boolean USE_CODES = false;

    static HashMap<String, String> acid_codes = new HashMap<>();

    public static void main(String[] args) {
        setAcidCodes();
        for (int i=0; i<5; i++) {
            FOLD_START = i;
            findSuccesses(true);
        }
    }

    public static void setAcidCodes(){
        acid_codes.put("A", "1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("C", "0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("D", "0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("E", "0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("F", "0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("G", "0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("H", "0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("I", "0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("K", "0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("L", "0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("M", "0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0");
        acid_codes.put("N", "0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0");
        acid_codes.put("P", "0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0");
        acid_codes.put("Q", "0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0");
        acid_codes.put("R", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0");
        acid_codes.put("S", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0");
        acid_codes.put("T", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0");
        acid_codes.put("V", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0");
        acid_codes.put("W", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0");
        acid_codes.put("Y", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0");
        acid_codes.put("B", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1");
        acid_codes.put("J", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1");
        acid_codes.put("O", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1");
        acid_codes.put("U", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1");
        acid_codes.put("X", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1");
        acid_codes.put("Z", "0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1");

    }
    static void findSuccesses(boolean successes){
        DateFormat dateFormat = new SimpleDateFormat("dd.MM.yy-HH.mm");
        Date date = new Date();
        String date_time = dateFormat.format(date);//.split(" ")[0];

        Random random = new Random(123456789);
        int strands = 0;
        int non_strands = 0;
        int proteins = 0;
        int acids = 0;
        int fail_length = 0;
        int test_length = 0;
        Set<String> success_proteins = new HashSet<String>();
        Set<String> failure_proteins = new HashSet<String>();
        Map<String, Integer> success_combinations = new HashMap<String, Integer>();

        String base_name = "pdb.11Dec2014.sorted_red25n.adataset";
        int file_number = FOLD_START;

        String contents = "";
        if(successes){
            contents = "success-set";
        }else {
            contents = "fail-set";
        }

        String range = "";
        if (FOLD_RANGE > 1){
            range = "_fold"+FOLD_START+"-"+(FOLD_RANGE+FOLD_START-1)+"_";
        }else {
            range = "_fold"+FOLD_START+"_";
        }
        try {
            BufferedWriter train_writer = new BufferedWriter(new FileWriter(date_time + range + WINDOW_SIZE +"window_train_" + contents +".txt"));
            BufferedWriter test_writer = new BufferedWriter(new FileWriter(date_time + range + WINDOW_SIZE +"window_test_" + contents +".txt"));
            BufferedWriter stat_writer = new BufferedWriter(new FileWriter(date_time + range + WINDOW_SIZE +"window_stats_" + contents +".txt"));
            BufferedWriter writer;
            for(int j=0; j<2; j++) {
                System.out.println("x");
                while (file_number < FOLD_START + FOLD_RANGE) {
                    String current_name = base_name + file_number++;
                    int line_count = 0;
                    Scanner scanner = new Scanner(new FileReader(current_name));
                    while (scanner.hasNextLine()) {
                        scanner.nextLine();
                        line_count++;
                    }
                    int training_length = (line_count / 5) * 4;
                    scanner = new Scanner(new FileReader(current_name));
                    int line_index = 0;
                    String protein_name = null;
                    String[] structure = null;
                    String[] events = null;
                    String[] combinations = null;
                    while (scanner.hasNextLine()) {
                        String next_line = scanner.nextLine();
                        line_index++;

                        if (line_index % 16 == 1) {
                            protein_name = next_line;
                        }
                        if (line_index % 16 == 3) {
                            structure = next_line.split("\t");
                        }
                        if (line_index % 16 == 4) {
                            events = next_line.split("\t");
                        }
                        if (line_index % 16 == 5) {
                            combinations = next_line.split("\t");
                            int current_middle_1 = WINDOW_SIZE / 2;
                            int current_middle_2 = current_middle_1 + 1;
                            proteins++;

                            while (current_middle_1 < (structure.length - WINDOW_SIZE / 2 - 1)) {
                                int output;

                                if (line_index < training_length) {
                                    writer = train_writer;
                                } else {
                                    writer = test_writer;
                                }

                                if (successes && events[current_middle_1].equals("E")) {
                                    if (Integer.parseInt(combinations[current_middle_1]) == current_middle_2 + 1) {
                                        strands++;
                                        output = 1;
                                        writer.write(protein_name + "\n");
                                        writer.write(1 + "\n");
                                        String chain_1 = "", chain_2 = "";
                                        for (int i = 0; i < WINDOW_SIZE; i++) {
                                            if (USE_CODES) {
                                                chain_1 += acid_codes.get(structure[current_middle_1 + i - WINDOW_SIZE / 2]) + " ";
                                                chain_2 += acid_codes.get(structure[current_middle_2 + i - WINDOW_SIZE / 2]) + " ";
                                            } else {
                                                chain_1 += structure[current_middle_1 + i - WINDOW_SIZE / 2] + " ";
                                                chain_2 += structure[current_middle_2 + i - WINDOW_SIZE / 2] + " ";
                                            }
                                        }
                                        String sequence = chain_1 + chain_2.trim() + "\n";
                                        writer.write(sequence);
                                        if (success_combinations.containsKey(sequence)) {
                                            success_combinations.replace(sequence, 1 + success_combinations.get(sequence));
                                        } else {
                                            success_combinations.put(sequence, 1);
                                        }
                                        if(line_index > training_length){
                                            test_length++;
                                        }
                                        writer.write(output + "\n");
                                        writer.write("\n");
                                    }
                                    if (!success_proteins.contains(protein_name)) {
                                        success_proteins.add(protein_name);
                                    }
                                } else if (!successes && !events[current_middle_1].equals("E")) {
                                    acids++;
                                    if (fail_length <= test_length && random.nextInt(3000) == 1) {
                                        if(line_index > training_length){
                                            fail_length++;
                                        }
                                        non_strands++;
                                        output = 0;
                                        writer.write(protein_name + "\n");
                                        writer.write(1 + "\n");
                                        String chain_1 = "", chain_2 = "";
                                        for (int i = 0; i < WINDOW_SIZE; i++) {
                                            if (USE_CODES) {
                                                chain_1 += acid_codes.get(structure[current_middle_1 + i - WINDOW_SIZE / 2]) + " ";
                                                chain_2 += acid_codes.get(structure[current_middle_2 + i - WINDOW_SIZE / 2]) + " ";
                                            } else {
                                                chain_1 += structure[current_middle_1 + i - WINDOW_SIZE / 2] + " ";
                                                chain_2 += structure[current_middle_2 + i - WINDOW_SIZE / 2] + " ";
                                            }
                                        }
                                        String sequence = chain_1 + chain_2.trim() + "\n";
                                        writer.write(sequence);
                                        if (success_combinations.containsKey(sequence)) {
                                            success_combinations.replace(sequence, 1 + success_combinations.get(sequence));
                                        } else {
                                            success_combinations.put(sequence, 1);
                                        }
                                        writer.write(output + "\n");
                                        writer.write("\n");
                                    }
                                } else if (!failure_proteins.contains(protein_name)) {
                                    failure_proteins.add(protein_name);
                                }
                                if (current_middle_2 == structure.length - 1 - WINDOW_SIZE / 2) {
                                    current_middle_1++;
                                    current_middle_2 = current_middle_1;
                                }
                                current_middle_2++;
                            }

                        }
                    }
                    while (scanner.hasNextLine() && !successes) {

                    }
                    scanner.close();
                }
                success_combinations = sortByComparator(success_combinations);
                success_combinations.forEach((k, v) -> System.out.println(k + ": " + v + " occurrence(s)"));
                System.out.println("success: " + success_proteins.size() + "\nfailure: " + failure_proteins.size());
                System.out.println(proteins +", " + strands +", "+ non_strands +", "+ acids);
                if(successes){
                    stat_writer.write("Success Set:\n\n");
                }else {
                    stat_writer.write("Fail Set:\n\n");
                }
                success_combinations.forEach((k, v) -> {
                    try {
                        stat_writer.write(k + ": " + v + " occurrence(s)\n");
                    }catch (java.io.IOException stat) {
                        System.out.println("Could not write to stat file");
                    }
                });
                successes = false;
                success_combinations = new HashMap<String, Integer>();
                file_number = FOLD_START;
            }

            train_writer.close();
            test_writer.close();
        }
        catch (FileNotFoundException fnfe) {

            System.out.println("File Not Found " + fnfe.toString());
        }
        catch (java.io.IOException uee) {
            System.out.println("Could not write to file");
        }

    }



    private static Map<String, Integer> sortByComparator(Map<String, Integer> unsortMap) {
        List<Entry<String, Integer>> list = new LinkedList<Entry<String, Integer>>(unsortMap.entrySet());

        Collections.sort(list, new Comparator<Entry<String, Integer>>() {
            public int compare(Entry<String, Integer> o1,
                               Entry<String, Integer> o2) {
                return o1.getValue().compareTo(o2.getValue());
            }
        });
        Map<String, Integer> sortedMap = new LinkedHashMap<String, Integer>();
        for (Entry<String, Integer> entry : list)
        {
            sortedMap.put(entry.getKey(), entry.getValue());
        }

        return sortedMap;
    }
}