package tk.gustavo.pages;


import org.fluentlenium.core.annotation.Page;
import org.fluentlenium.core.annotation.PageUrl;
import org.fluentlenium.core.domain.FluentWebElement;
import org.openqa.selenium.support.FindBy;

@PageUrl("http://localhost:5000/register")
public class RegisterPage extends BasePage {

    @Page
    private LoginPage loginPage;

    @FindBy(id = "username")
    private FluentWebElement usernameInput;

    @FindBy(id = "email")
    private FluentWebElement emailInput;

    @FindBy(id = "password")
    private FluentWebElement passwordInput;

    @FindBy(id = "confirm_password")
    private FluentWebElement confirmPasswordInput;

    @FindBy(id = "submit")
    private FluentWebElement submitButton;

    public RegisterPage fillUsername(String username){
        usernameInput.clear().write(username);
        return this;
    }

    public RegisterPage fillEmail(String email){
        emailInput.clear().write(email);
        return this;
    }

    public RegisterPage fillPassword(String password){
        passwordInput.clear().write(password);
        return this;
    }

    public RegisterPage fillconfirmPassword(String password){
        confirmPasswordInput.clear().write(password);
        return this;
    }

    public RegisterPage fillBothPasswordFields(String password){
        passwordInput.clear().write(password);
        confirmPasswordInput.clear().write(password);
        return this;
    }

    public LoginPage clickSignInButton(){
        submitButton.submit();
        return loginPage;
    }

    public LoginPage register(String username, String email, String password){
        return fillUsername(username)
                .fillEmail(email)
                .fillPassword(password)
                .fillconfirmPassword(password)
                .clickSignInButton();
    }



}
