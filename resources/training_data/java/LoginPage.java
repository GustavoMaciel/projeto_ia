package tk.gustavo.pages;

import org.fluentlenium.core.annotation.PageUrl;
import org.fluentlenium.core.domain.FluentWebElement;
import org.openqa.selenium.support.FindBy;

@PageUrl("http://localhost:5000/login")
public class LoginPage extends BasePage {

    @FindBy(name = "email")
    private FluentWebElement emailInput;

    @FindBy(name = "password")
    private FluentWebElement passwordInput;

    @FindBy(name = "submit")
    private FluentWebElement submitButton;


    public LoginPage fillEmail(String email){
        emailInput.clear().write(email);
        return this;
    }

    public LoginPage fillPassword(String password){
        passwordInput.clear().write(password);
        return this;
    }

    public HomePage clickSubmitButton(){
        submitButton.submit();
        return homePage;
    }

    public HomePage login(String email, String password){
        return fillEmail(email).fillPassword(password).clickSubmitButton();
    }


}
