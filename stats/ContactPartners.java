//Object to store information relevant to the final neural network input before file write
public class ContactPartners {
    String sequence;
    String id;
    String classification;
    ContactPartners(String sequence, String id, String classification){
        this.sequence = sequence;
        this.id = id;
        this.classification = classification;
    }
}
