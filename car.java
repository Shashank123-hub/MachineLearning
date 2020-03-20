package abstraction;

abstract class car {
  public abstract void carCompany();
  public abstract void carFeatures();
  public void carColor(){
  System.out.println("The car color is black.");
}
  public void carMake() {
	System.out.println("It was manufactured in 2017.");
  }
}

class tesla extends car{
	public void carCompany() {
		System.out.println("The car company is Tesla");
	}
	public void carFeatures() {
		System.out.println("The car has an auto-piloting system.");
	}
}

class driverClass{
	public static void main(String[] args) {
		tesla carObj = new tesla(); //Since abstract class cannot be called by an object, we crate an object for subclass.
		carObj.carCompany();
		carObj.carFeatures();
		carObj.carColor();
		carObj.carMake();
	}
}